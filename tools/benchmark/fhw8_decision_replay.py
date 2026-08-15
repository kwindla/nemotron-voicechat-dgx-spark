#!/usr/bin/env python3
"""fhw8 qualification drivers: G1 decision replay and G0(b) head numerics.

Subcommands:
  replay    Replay the frozen capture on one artifact lane, recording the
            per-call text token and function token, each checked against the
            retained capture's decisions (basin + decision contract).
  compare   G1 comparator over one baseline-lane and one candidate-lane
            replay output: zero per-call decision flips across all valid
            blocks, both lanes matching the retained capture.
  g0b       Head-numerics gate on dumped final hidden states: F32-reference
            logits vs candidate-dequant logits; argmax agreement >= 99.9%
            and preserved top-2 ordering wherever the reference margin
            exceeds 0.25 (preregistration G0b).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

CALL39_DIVERGENT_SIGNATURE = {"call_index": 39, "expected": 10592, "actual": 1044}

HARNESS_PATH = Path(__file__).resolve().parent / "nano_interface_decomposition.py"
_spec = importlib.util.spec_from_file_location("nano_interface_decomposition", HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("nano_interface_decomposition", harness)
_spec.loader.exec_module(harness)


def atomic_json(path: Path, payload: Any) -> None:
    harness.atomic_json(path, payload)


def _load_model_mode(args: argparse.Namespace):
    """harness._load_model with an eager override for the extraction lane."""
    import os
    import sys as _sys

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        }
    )
    _sys.path.insert(0, str(args.speech_root.resolve()))
    from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model

    return create_model(
        engine_type="vllm_llm",
        vllm_config={
            "model_path": str(args.nano_vllm_path.resolve()),
            "engine_path": str(args.nano_vllm_path.resolve()),
            "max_model_len": int(args.max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "dtype": "bfloat16",
            "enforce_eager": bool(args.enforce_eager),
        },
        top_p=1.0,
        repetition_penalty=1.0,
        temperature=0.0,
        text_pad_id=harness.PAD_TOKEN_ID,
    )


def replay(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    root = args.capture_root.resolve()
    records = harness.read_manifest(root)
    model = _load_model_mode(args)
    blocks: list[dict[str, Any]] = []
    try:
        for ordinal in range(args.warmup_blocks + args.measured_blocks):
            measured = ordinal >= args.warmup_blocks
            block_index = ordinal - args.warmup_blocks if measured else ordinal
            phase = "measured" if measured else "warmup"
            request_id = f"fhw8-{args.lane}-{phase}-{block_index:03d}"
            decisions: list[dict[str, int]] = []
            text_mismatches: list[dict[str, int]] = []
            function_mismatches: list[dict[str, int]] = []
            started = time.perf_counter()
            cleanup_ok = False
            try:
                for record in records:
                    payload = torch.load(
                        root / record["file"], map_location="cpu", weights_only=False
                    )
                    saved = payload["inputs"]
                    call_index = int(record["call_index"])
                    kwargs = {
                        "current_step": int(saved.get("current_step", 0)),
                        "decode_steps": int(saved.get("decode_steps", 1)),
                    }
                    if saved.get("prompt_token_ids") is not None:
                        kwargs["prompt_token_ids"] = saved["prompt_token_ids"]
                    generated = saved.get("generated_tokens")
                    if generated is not None:
                        kwargs["generated_tokens"] = generated.to("cuda")
                    result = model(
                        saved["input_embeds"].to("cuda"), request_id=request_id, **kwargs
                    )
                    text_token = int(result["predicted_token"])
                    function_value = result.get("function_predicted_token")
                    function_token = (
                        int(torch.as_tensor(function_value).reshape(-1)[0])
                        if function_value is not None
                        else -1
                    )
                    retained_text = int(record["predicted_token"])
                    retained_function_value = payload["outputs"]["interface_result"].get(
                        "function_predicted_token"
                    )
                    retained_function = (
                        int(torch.as_tensor(retained_function_value).reshape(-1)[0])
                        if retained_function_value is not None
                        else -1
                    )
                    decisions.append(
                        {
                            "call_index": call_index,
                            "text_token": text_token,
                            "function_token": function_token,
                        }
                    )
                    if text_token != retained_text:
                        text_mismatches.append(
                            {
                                "call_index": call_index,
                                "expected": retained_text,
                                "actual": text_token,
                            }
                        )
                    if function_token != retained_function:
                        function_mismatches.append(
                            {
                                "call_index": call_index,
                                "expected": retained_function,
                                "actual": function_token,
                            }
                        )
                model.abort_request(request_id)
                cleanup_ok = True
            finally:
                if not cleanup_ok:
                    try:
                        model.abort_request(request_id)
                    except Exception:
                        pass
            blocks.append(
                {
                    "block": block_index,
                    "phase": phase,
                    "calls": len(decisions),
                    "wall_ms": (time.perf_counter() - started) * 1000.0,
                    "text_mismatches": text_mismatches,
                    "function_mismatches": function_mismatches,
                    "valid": measured
                    and not text_mismatches
                    and not function_mismatches
                    and len(decisions) == len(records),
                    "decisions": decisions if measured else None,
                }
            )
    finally:
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    measured_blocks = [b for b in blocks if b["phase"] == "measured"]
    valid_blocks = [b for b in measured_blocks if b["valid"]]
    payload = {
        "kind": "fhw8_decision_replay",
        "lane": args.lane,
        "enforce_eager": bool(args.enforce_eager),
        "artifact": str(args.nano_vllm_path.resolve()),
        "capture_manifest_sha256": harness.sha256_file(root / "manifest.jsonl"),
        "warmup_blocks": args.warmup_blocks,
        "measured_blocks": len(measured_blocks),
        "valid_blocks": len(valid_blocks),
        "blocks": blocks,
        "passed": len(valid_blocks) >= args.min_valid_blocks,
    }
    return payload


def _block_valid_for_basin(block: dict[str, Any], basin: str) -> bool:
    if block.get("phase") != "measured" or block.get("function_mismatches"):
        return False
    text = block.get("text_mismatches") or []
    if basin == "exact":
        return not text
    # call39-divergent: exactly the characterized Step 5 signature, nothing else
    return text == [CALL39_DIVERGENT_SIGNATURE]


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    flips: list[dict[str, Any]] = []
    pairs_checked = 0
    if args.basin == "as-run":
        baseline_valid = [b for b in baseline["blocks"] if b.get("valid")]
        candidate_valid = [b for b in candidate["blocks"] if b.get("valid")]
    else:
        baseline_valid = [
            b for b in baseline["blocks"] if _block_valid_for_basin(b, args.basin)
        ]
        candidate_valid = [
            b for b in candidate["blocks"] if _block_valid_for_basin(b, args.basin)
        ]
    for base_block in baseline_valid:
        base_map = {
            d["call_index"]: (d["text_token"], d["function_token"])
            for d in base_block["decisions"]
        }
        for cand_block in candidate_valid:
            for decision in cand_block["decisions"]:
                expected = base_map[decision["call_index"]]
                actual = (decision["text_token"], decision["function_token"])
                pairs_checked += 1
                if actual != expected:
                    flips.append(
                        {
                            "baseline_block": base_block["block"],
                            "candidate_block": cand_block["block"],
                            "call_index": decision["call_index"],
                            "baseline": expected,
                            "candidate": actual,
                        }
                    )
    payload = {
        "kind": "fhw8_g1_comparison",
        "basin": args.basin,
        "baseline_valid_blocks": len(baseline_valid),
        "candidate_valid_blocks": len(candidate_valid),
        "min_valid_blocks": args.min_valid_blocks,
        "decision_pairs_checked": pairs_checked,
        "decision_flips": flips,
        "passed": (
            len(baseline_valid) >= args.min_valid_blocks
            and len(candidate_valid) >= args.min_valid_blocks
            and not flips
            and (
                args.basin != "as-run" or (baseline["passed"] and candidate["passed"])
            )
        ),
    }
    return payload


def g0b(args: argparse.Namespace) -> dict[str, Any]:
    import numpy
    import torch

    build_spec = importlib.util.spec_from_file_location(
        "build_fhw8_candidate", Path(__file__).resolve().parent / "build_fhw8_candidate.py"
    )
    builder = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(builder)
    from safetensors.torch import load_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_chunks = []
    for dump in sorted(args.hidden_dir.glob("hidden-*.bin")):
        raw = numpy.fromfile(dump, dtype=numpy.float32)
        if raw.size % 4480:
            raise ValueError(f"{dump} is not a whole number of 4480-dim vectors")
        hidden_chunks.append(torch.from_numpy(raw.reshape(-1, 4480)))
    if not hidden_chunks:
        raise ValueError(f"no hidden-state dumps under {args.hidden_dir}")
    hidden = torch.cat(hidden_chunks)
    total_dumped = hidden.shape[0]
    if args.take_last and hidden.shape[0] > args.take_last:
        # engine-init compile/warmup dummies precede the capture replay in the
        # dump stream; only the trailing capture-replay vectors are used
        hidden = hidden[-args.take_last :]
    if hidden.shape[0] < args.min_vectors:
        raise ValueError(f"only {hidden.shape[0]} vectors; need >= {args.min_vectors}")

    release_index = json.loads(
        (args.release_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    fh_shard = release_index["weight_map"][builder.FUNCTION_HEAD]
    w_f32 = load_file(args.release_root / fh_shard)[builder.FUNCTION_HEAD]
    cand_index = json.loads(
        (args.candidate_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    q_shard = cand_index["weight_map"]["stt_model.function_head.qweight"]
    cand_tensors = load_file(args.candidate_root / q_shard)
    packed = {
        name: cand_tensors[f"stt_model.function_head.{name}"] for name in builder.Q_NAMES
    }
    w_deq = builder.dequantize(packed, w_f32.shape[0])

    w_ref = w_f32.to(device=device, dtype=torch.float32)
    w_cand = w_deq.to(device=device, dtype=torch.float32)
    agree = 0
    margin_checked = 0
    margin_violations = 0
    total = hidden.shape[0]
    for start in range(0, total, args.batch):
        h = hidden[start : start + args.batch].to(device=device, dtype=torch.float32)
        ref = h @ w_ref.T
        cand = h @ w_cand.T
        top2 = ref.topk(2, dim=1)
        ref_arg = top2.indices[:, 0]
        cand_arg = cand.argmax(dim=1)
        agree += int((ref_arg == cand_arg).sum())
        margin = top2.values[:, 0] - top2.values[:, 1]
        strong = margin > args.margin
        margin_checked += int(strong.sum())
        rows = torch.arange(h.shape[0], device=device)
        cand_gap = cand[rows, top2.indices[:, 0]] - cand[rows, top2.indices[:, 1]]
        margin_violations += int((strong & (cand_gap <= 0)).sum())
    agreement = agree / total
    payload = {
        "kind": "fhw8_g0b_head_numerics",
        "vectors_dumped_total": total_dumped,
        "vectors": total,
        "argmax_agreement": agreement,
        "argmax_agreement_required": 0.999,
        "strong_margin_threshold": args.margin,
        "strong_margin_checked": margin_checked,
        "strong_margin_violations": margin_violations,
        "passed": agreement >= 0.999 and margin_violations == 0,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rp = sub.add_parser("replay")
    rp.add_argument("--lane", required=True, choices=("baseline", "candidate"))
    rp.add_argument("--speech-root", type=Path, required=True)
    rp.add_argument("--nano-vllm-path", type=Path, required=True)
    rp.add_argument("--capture-root", type=Path, required=True)
    rp.add_argument("--warmup-blocks", type=int, default=2)
    rp.add_argument("--enforce-eager", action="store_true")
    rp.add_argument("--measured-blocks", type=int, default=15)
    rp.add_argument("--min-valid-blocks", type=int, default=3)
    rp.add_argument("--max-model-len", type=int, default=12288)
    rp.add_argument("--gpu-memory-utilization", type=float, default=0.42)
    rp.add_argument("--output", type=Path, required=True)

    cp = sub.add_parser("compare")
    cp.add_argument("--baseline", type=Path, required=True)
    cp.add_argument("--candidate", type=Path, required=True)
    cp.add_argument("--min-valid-blocks", type=int, default=3)
    cp.add_argument("--basin", choices=("as-run", "exact", "call39-divergent"), default="as-run")
    cp.add_argument("--output", type=Path, required=True)

    gb = sub.add_parser("g0b")
    gb.add_argument("--hidden-dir", type=Path, required=True)
    gb.add_argument("--release-root", type=Path, required=True)
    gb.add_argument("--candidate-root", type=Path, required=True)
    gb.add_argument("--min-vectors", type=int, default=10000)
    gb.add_argument("--take-last", type=int, default=0)
    gb.add_argument("--margin", type=float, default=0.25)
    gb.add_argument("--batch", type=int, default=256)
    gb.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "replay":
        payload = replay(args)
    elif args.command == "compare":
        payload = compare(args)
    else:
        payload = g0b(args)
    atomic_json(args.output, payload)
    condensed = {
        k: v for k, v in payload.items() if k not in {"blocks", "decision_flips"}
    }
    condensed["decision_flip_count"] = len(payload.get("decision_flips", []))
    print(json.dumps(condensed, indent=2, sort_keys=True, default=str))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
