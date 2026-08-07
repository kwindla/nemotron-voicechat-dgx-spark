#!/usr/bin/env python3
"""Same-process exact replay gate for the Step-7 Nano pair graph.

The harness keeps two independent requests in one vLLM worker.  Request names
select the unmodified PIECEWISE baseline or the dedicated query_len=2 FULL
graph.  Captured fused inputs and causal text prefixes are applied in lockstep;
timings from this deliberately synchronized diagnostic are invalid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nano-vllm-path", type=Path, default=Path("/models/derived/nano"))
    parser.add_argument("--speech-root", type=Path, default=Path("/opt/voicechat"))
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    return parser.parse_args()


def _normalize(value: Any, *, request_id: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            str(key): ("<request>" if key == "request_id" else _normalize(item, request_id=request_id))
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_normalize(item, request_id=request_id) for item in value)
    if isinstance(value, list):
        return [_normalize(item, request_id=request_id) for item in value]
    if value == request_id:
        return "<request>"
    return value


def _exact(left: Any, right: Any, path: str = "root") -> str | None:
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return f"{path}: tensor/type mismatch {type(left)!r} != {type(right)!r}"
        if left.dtype != right.dtype or left.shape != right.shape:
            return (
                f"{path}: tensor metadata mismatch "
                f"{left.dtype}/{tuple(left.shape)} != {right.dtype}/{tuple(right.shape)}"
            )
        if not torch.equal(left, right):
            count = int(torch.count_nonzero(left != right))
            return f"{path}: {count}/{left.numel()} tensor elements differ"
        return None
    if type(left) is not type(right):
        return f"{path}: type mismatch {type(left)!r} != {type(right)!r}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys differ {sorted(left)} != {sorted(right)}"
        for key in left:
            mismatch = _exact(left[key], right[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path}: lengths differ {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatch = _exact(left_item, right_item, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def read_request_cache_results(worker: Any) -> list[dict[str, Any]]:
    """collective_rpc entry point: fetch in-worker post-rollback comparisons."""
    return list(getattr(worker.model_runner, "_voicechat_step7_cache_results", []))


def _decision_view(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "buffered",
        "accepted",
        "rejected",
        "sequential_bypass",
        "sequential_pending_drains",
        "conditional_singles",
        "sequential_mode",
        "previous_effective_pad",
        "control_barriers",
        "control_pending_drains",
        "precontrol_barriers",
        "precontrol_pending_drains",
        "precontrol_reasons",
        "needs_correction",
        "assumed",
    )
    return {key: state.get(key) for key in keys} | {"pending": state.get("pending") is not None}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VOICECHAT_NANO_PAD_PAIR": "1",
            "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1",
            "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "1",
            "VOICECHAT_NANO_PAIR_FULL_GRAPH": "1",
            "VOICECHAT_STEP7_BITWISE_GATE": "1",
            # The only pickled callable is compare_request_caches above.  It
            # executes locally in this isolated diagnostic container.
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        }
    )
    sys.path.insert(0, str(args.speech_root.resolve()))

    import torch
    from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model
    from nemotron_voicechat_runtime.runtime_optimizations import (
        _install_public_pad_pair_engine,
        _pad_pair_state,
    )

    _install_public_pad_pair_engine()
    records = [
        json.loads(line)
        for line in (args.teacher_replay / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.max_calls:
        records = records[: args.max_calls]
    if not records:
        raise SystemExit("teacher replay is empty")

    model = create_model(
        engine_type="vllm_llm",
        vllm_config={
            "model_path": str(args.nano_vllm_path),
            "engine_path": str(args.nano_vllm_path),
            "max_model_len": 12288,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "bfloat16",
            "enforce_eager": False,
        },
        top_p=1.0,
        repetition_penalty=1.0,
        temperature=0.0,
        text_pad_id=12,
    )
    old_request = "step7-old-silent-turn"
    new_request = "step7-new-silent-turn"
    comparisons: list[dict[str, Any]] = []
    mismatch: dict[str, Any] | None = None
    cache_checks = 0
    last_old_function = 12
    last_new_function = 12
    try:
        for record in records:
            payload = torch.load(
                args.teacher_replay / record["file"], map_location="cpu", weights_only=False
            )
            saved = payload["inputs"]
            current_step = int(saved.get("current_step", 0))
            kwargs: dict[str, Any] = {
                "current_step": current_step,
                "decode_steps": int(saved.get("decode_steps", 1)),
            }
            if saved.get("prompt_token_ids") is not None:
                kwargs["prompt_token_ids"] = saved["prompt_token_ids"]
            generated = saved.get("generated_tokens")
            if generated is not None:
                kwargs["generated_tokens"] = generated.to("cuda")

            # Reconstruct the conditional pair input from the captured causal
            # text prefix.  Function state evolves independently in each lane
            # and is required to match after every call.
            if current_step > 0 and generated is not None and generated.numel() >= current_step:
                previous_text = int(generated.reshape(-1)[current_step - 1])
                _pad_pair_state(model.engine, old_request)["previous_effective_pad"] = bool(
                    previous_text == 12 and last_old_function == 12
                )
                _pad_pair_state(model.engine, new_request)["previous_effective_pad"] = bool(
                    previous_text == 12 and last_new_function == 12
                )

            old_result = model(
                saved["input_embeds"].to("cuda"), request_id=old_request, **kwargs
            )
            new_result = model(
                saved["input_embeds"].to("cuda"), request_id=new_request, **kwargs
            )
            old_normalized = _normalize(old_result, request_id=old_request)
            new_normalized = _normalize(new_result, request_id=new_request)
            output_mismatch = _exact(old_normalized, new_normalized)
            old_state = _pad_pair_state(model.engine, old_request)
            new_state = _pad_pair_state(model.engine, new_request)
            decision_mismatch = _exact(_decision_view(old_state), _decision_view(new_state))

            if isinstance(old_result, dict) and old_result.get("function_predicted_token") is not None:
                last_old_function = int(old_result["function_predicted_token"].reshape(-1)[-1])
            if isinstance(new_result, dict) and new_result.get("function_predicted_token") is not None:
                last_new_function = int(new_result["function_predicted_token"].reshape(-1)[-1])

            decisions = _decision_view(old_state)
            executed = sum(
                int(decisions[key])
                for key in ("accepted", "rejected", "conditional_singles")
            )
            pair_executed = int(decisions["accepted"]) + int(decisions["rejected"])
            previous_pair_executed = (
                comparisons[-1]["pair_executed"] if comparisons else 0
            )
            cache_result = None
            if pair_executed != previous_pair_executed:
                cache_results = model._run_async(
                    model.engine.engine.collective_rpc(
                        read_request_cache_results,
                    )
                )[0]
                if len(cache_results) != pair_executed:
                    cache_result = {
                        "equal": False,
                        "mismatch": (
                            f"worker returned {len(cache_results)} cache checks "
                            f"after {pair_executed} pair executions"
                        ),
                    }
                else:
                    cache_result = cache_results[-1]
                cache_checks += 1

            item = {
                "call_index": int(record["call_index"]),
                "current_step": current_step,
                "output_equal": output_mismatch is None,
                "decision_equal": decision_mismatch is None,
                "cache": cache_result,
                "decisions": decisions,
                "executed": executed,
                "pair_executed": pair_executed,
            }
            comparisons.append(item)
            if output_mismatch or decision_mismatch or (
                cache_result is not None and not cache_result.get("equal", False)
            ):
                mismatch = item | {
                    "output_mismatch": output_mismatch,
                    "decision_mismatch": decision_mismatch,
                }
                break
    finally:
        for request_id in (old_request, new_request):
            try:
                model.abort_request(request_id)
            except Exception:
                pass
        model.shutdown()

    final_decisions = comparisons[-1]["decisions"] if comparisons else {}
    passed = mismatch is None and len(comparisons) == len(records)
    report = {
        "schema": 1,
        "kind": "step7_same_process_pair_graph_bitwise_gate",
        "passed": passed,
        "warning": "Diagnostic synchronization invalidates all timings.",
        "teacher_replay": str(args.teacher_replay.resolve()),
        "requested_calls": len(records),
        "completed_calls": len(comparisons),
        "cache_checks": cache_checks,
        "baseline_request": old_request,
        "candidate_request": new_request,
        "baseline_pair_mode": "PIECEWISE",
        "candidate_pair_mode": "FULL query_len=2",
        "final_decisions": final_decisions,
        "first_mismatch": mismatch,
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("Step-7 same-process bitwise gate failed")


if __name__ == "__main__":
    main()
