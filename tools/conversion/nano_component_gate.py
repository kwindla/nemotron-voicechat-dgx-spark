#!/usr/bin/env python3
"""Replay exact public Nano embeddings and localize teacher/candidate drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from nano_attribution import (
    MODES as ATTRIBUTION_MODES,
)
from nano_attribution import (
    compare_snapshot_directories,
    finalize_nano_attribution,
    install_nano_attribution,
    parse_int_set,
    target_window_summary,
)
from nano_replay import CapturingNanoInterface, NanoReplayRecorder, read_manifest

PRODUCTION_CUSTOM_OUTPUTS = ["function_tokens", "function_logits"]
DIAGNOSTIC_CUSTOM_OUTPUTS = ["text_logits", *PRODUCTION_CUSTOM_OUTPUTS]
SKIP_TEXT_LOGITS_CONTRACT = "temperature=0, top_p=1, repetition_penalty=1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_diagnostic_candidate(source: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Expose text logits for the gate without changing the release artifact.

    Production Candidate 1 intentionally omits the full-vocabulary text-logit
    custom output under a deterministic-sampling contract. The control-margin
    gate still needs those logits. For that one diagnostic run, construct a
    symlinked, weight-identical view and change only the declared custom-output
    list. Arbitrary artifacts that omit text logits remain a hard failure.
    """

    source = source.expanduser().resolve()
    source_config_path = source / "config.json"
    config = json.loads(source_config_path.read_text(encoding="utf-8"))
    original_outputs = config.get("custom_outputs")
    source_config_sha256 = sha256_file(source_config_path)
    if original_outputs == DIAGNOSTIC_CUSTOM_OUTPUTS:
        return source, {
            "mode": "artifact-native-text-logits",
            "source_config_sha256": source_config_sha256,
            "diagnostic_config_sha256": source_config_sha256,
            "weights_changed": False,
        }

    skip_contract = config.get("voicechat_skip_custom_text_logits")
    if original_outputs != PRODUCTION_CUSTOM_OUTPUTS or skip_contract != {
        "contract": SKIP_TEXT_LOGITS_CONTRACT,
        "enabled": True,
        "source_config_sha256": skip_contract.get("source_config_sha256")
        if isinstance(skip_contract, dict)
        else None,
    }:
        raise ValueError(
            "candidate omits text_logits without the qualified deterministic-sampling contract"
        )
    if not skip_contract["source_config_sha256"]:
        raise ValueError("candidate text-logit skip contract has no source config digest")

    diagnostic = output_dir / "diagnostic-candidate"
    if diagnostic.exists():
        shutil.rmtree(diagnostic)
    diagnostic.mkdir(parents=True)
    for entry in sorted(source.iterdir()):
        if not entry.is_file():
            raise ValueError(f"unexpected non-file in Nano artifact: {entry}")
        if entry.name != "config.json":
            (diagnostic / entry.name).symlink_to(entry)
    diagnostic_config = dict(config)
    diagnostic_config["custom_outputs"] = DIAGNOSTIC_CUSTOM_OUTPUTS
    diagnostic_config.pop("voicechat_skip_custom_text_logits")
    diagnostic_config_path = diagnostic / "config.json"
    diagnostic_config_path.write_text(
        json.dumps(diagnostic_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return diagnostic, {
        "mode": "temporary-text-logit-overlay",
        "source_config_sha256": source_config_sha256,
        "diagnostic_config_sha256": sha256_file(diagnostic_config_path),
        "source_custom_outputs": original_outputs,
        "diagnostic_custom_outputs": DIAGNOSTIC_CUSTOM_OUTPUTS,
        "sampling_contract": skip_contract["contract"],
        "weights_changed": False,
        "weight_files": sum(1 for entry in diagnostic.iterdir() if entry.name != "config.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--nano-vllm-path", type=Path, required=True)
    parser.add_argument("--teacher-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-calls", type=int, default=512)
    parser.add_argument("--require-token-agreement", type=float, default=1.0)
    parser.add_argument("--min-logit-cosine", type=float, default=0.995)
    parser.add_argument("--bos-token-id", type=int, default=1)
    parser.add_argument("--pad-token-id", type=int, default=12)
    parser.add_argument("--max-bos-pad-margin-delta", type=float, default=1.0)
    parser.add_argument("--require-bos-pad-order-agreement", type=float, default=1.0)
    parser.add_argument(
        "--allow-missing-control-margins",
        action="store_true",
        help=(
            "Permit token-only diagnostic replay when the teacher artifact deliberately "
            "omits text_logits. The default qualification gate remains strict."
        ),
    )
    parser.add_argument(
        "--attribution-mode", choices=("off", *sorted(ATTRIBUTION_MODES)), default="off"
    )
    parser.add_argument("--attribution-layers", default="33,35,37,40,42,45,47,49,51,53")
    parser.add_argument("--attribution-windows", default="20-30,50-55")
    parser.add_argument("--attribution-output-dir", type=Path)
    parser.add_argument("--attribution-reference-dir", type=Path)
    parser.add_argument("--attribution-teacher-weight-file", type=Path)
    return parser.parse_args()


def first_engine_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    results = payload["outputs"].get("engine_results") or []
    return results[-1] if results else None


def text_logits(payload: dict[str, Any]):
    result = first_engine_result(payload)
    if not result:
        return None
    return (result.get("custom_outputs") or {}).get("text_logits")


def scalar_token(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "numel"):
        if value.numel() != 1:
            raise ValueError(f"expected scalar token, got {tuple(value.shape)}")
        return int(value.item())
    return int(value)


def control_margin(logits, *, bos_token_id: int, pad_token_id: int) -> float | None:
    if logits is None:
        return None
    flat = logits.float().reshape(-1, logits.shape[-1])[-1]
    if min(bos_token_id, pad_token_id) < 0 or max(bos_token_id, pad_token_id) >= flat.numel():
        raise ValueError(
            f"BOS/PAD token IDs {(bos_token_id, pad_token_id)} exceed logits size {flat.numel()}"
        )
    return float(flat[bos_token_id] - flat[pad_token_id])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.teacher_replay)[: args.max_calls]
    if not records:
        raise SystemExit("teacher replay is empty")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        }
    )
    sys.path.insert(0, str(args.speech_root.resolve()))

    import torch
    from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model

    candidate_model_path, diagnostic_config = prepare_diagnostic_candidate(
        args.nano_vllm_path, args.output_dir
    )
    candidate_capture = args.output_dir / "candidate-capture"
    recorder = NanoReplayRecorder(candidate_capture, max_calls=len(records))
    model = create_model(
        engine_type="vllm_llm",
        vllm_config={
            "model_path": str(candidate_model_path),
            "engine_path": str(candidate_model_path),
            "max_model_len": 6144,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "bfloat16",
            "enforce_eager": bool(args.enforce_eager),
        },
        top_p=1.0,
        repetition_penalty=1.0,
        temperature=0.0,
        text_pad_id=12,
    )
    attribution_layers = parse_int_set(args.attribution_layers)
    attribution_windows = parse_int_set(args.attribution_windows)
    attribution_output = args.attribution_output_dir or (args.output_dir / "attribution")
    attribution_worker_result = None
    attribution_finalize_result = None
    try:
        if args.attribution_mode != "off":
            if not args.enforce_eager:
                raise SystemExit("Nano attribution is profiling-only and requires --enforce-eager")
            attribution_output.mkdir(parents=True, exist_ok=True)
            worker_config = {
                "mode": args.attribution_mode,
                "layers": sorted(attribution_layers),
                "windows": sorted(attribution_windows),
                "output_dir": str(attribution_output.resolve()),
                "reference_dir": (
                    str(args.attribution_reference_dir.resolve())
                    if args.attribution_reference_dir
                    else None
                ),
                "teacher_weight_file": (
                    str(args.attribution_teacher_weight_file.resolve())
                    if args.attribution_teacher_weight_file
                    else None
                ),
            }
            attribution_worker_result = model._run_async(
                model.engine.engine.collective_rpc(
                    install_nano_attribution, kwargs={"config": worker_config}
                )
            )
    except BaseException:
        model.shutdown()
        raise
    captured = CapturingNanoInterface(model, recorder)
    request_ids: dict[tuple[int, str], str] = {}
    call_times: list[float] = []
    try:
        for record in records:
            teacher = torch.load(
                args.teacher_replay / record["file"], map_location="cpu", weights_only=False
            )
            saved = teacher["inputs"]
            teacher_request = str(saved.get("request_id", "default"))
            request_key = (int(record.get("sequence_epoch", 0)), teacher_request)
            request_id = request_ids.setdefault(
                request_key, f"nano-component-replay-{len(request_ids)}"
            )
            kwargs = {
                "current_step": int(saved.get("current_step", 0)),
                "decode_steps": int(saved.get("decode_steps", 1)),
            }
            if saved.get("prompt_token_ids") is not None:
                kwargs["prompt_token_ids"] = saved["prompt_token_ids"]
            generated = saved.get("generated_tokens")
            if generated is not None:
                kwargs["generated_tokens"] = generated.to("cuda")
            started = time.perf_counter()
            captured(
                saved["input_embeds"].to("cuda"),
                request_id=request_id,
                **kwargs,
            )
            torch.cuda.synchronize()
            call_times.append(time.perf_counter() - started)
    finally:
        try:
            for request_id in request_ids.values():
                model.abort_request(request_id)
        finally:
            try:
                if args.attribution_mode != "off":
                    attribution_finalize_result = model._run_async(
                        model.engine.engine.collective_rpc(finalize_nano_attribution)
                    )
            finally:
                model.shutdown()

    candidate_records = read_manifest(candidate_capture)
    comparisons = []
    agreements = 0
    cosine_values = []
    control_margin_deltas = []
    control_order_agreements = 0
    control_margin_pairs = []
    first_control_order_divergence = None
    first_divergence = None
    for teacher_record, candidate_record in zip(records, candidate_records, strict=True):
        teacher = torch.load(
            args.teacher_replay / teacher_record["file"], map_location="cpu", weights_only=False
        )
        candidate = torch.load(
            candidate_capture / candidate_record["file"], map_location="cpu", weights_only=False
        )
        teacher_token = scalar_token(teacher["outputs"].get("predicted_token"))
        candidate_token = scalar_token(candidate["outputs"].get("predicted_token"))
        agrees = teacher_token == candidate_token
        agreements += int(agrees)
        left, right = text_logits(teacher), text_logits(candidate)
        cosine = None
        max_abs = None
        if left is not None and right is not None:
            left = left.float().reshape(-1)
            right = right.float().reshape(-1)
            cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
            max_abs = float((left - right).abs().max())
            cosine_values.append(cosine)
        teacher_margin = control_margin(
            left,
            bos_token_id=args.bos_token_id,
            pad_token_id=args.pad_token_id,
        )
        candidate_margin = control_margin(
            right,
            bos_token_id=args.bos_token_id,
            pad_token_id=args.pad_token_id,
        )
        margin_delta = None
        order_agrees = None
        if teacher_margin is not None and candidate_margin is not None:
            margin_delta = candidate_margin - teacher_margin
            order_agrees = (teacher_margin >= 0) == (candidate_margin >= 0)
            control_order_agreements += int(order_agrees)
            control_margin_deltas.append(margin_delta)
            control_margin_pairs.append((teacher_margin, candidate_margin))
        item = {
            "call_index": teacher_record["call_index"],
            "current_step": teacher_record.get("current_step"),
            "teacher_token": teacher_token,
            "candidate_token": candidate_token,
            "token_agrees": agrees,
            "logit_cosine": cosine,
            "logit_max_abs": max_abs,
            "teacher_bos_minus_pad_margin": teacher_margin,
            "candidate_bos_minus_pad_margin": candidate_margin,
            "bos_pad_margin_delta": margin_delta,
            "bos_pad_order_agrees": order_agrees,
        }
        comparisons.append(item)
        if first_divergence is None and not agrees:
            first_divergence = item
        if first_control_order_divergence is None and order_agrees is False:
            first_control_order_divergence = item

    token_agreement = agreements / len(comparisons)
    min_cosine = min(cosine_values) if cosine_values else None
    control_order_agreement = (
        control_order_agreements / len(control_margin_pairs) if control_margin_pairs else 0.0
    )
    max_control_margin_delta = (
        max(abs(value) for value in control_margin_deltas) if control_margin_deltas else None
    )
    control_gate_passed = (args.allow_missing_control_margins and not control_margin_pairs) or (
        max_control_margin_delta is not None
        and max_control_margin_delta <= args.max_bos_pad_margin_delta
        and control_order_agreement >= args.require_bos_pad_order_agreement
    )
    passed = (
        token_agreement >= args.require_token_agreement
        and (min_cosine is None or min_cosine >= args.min_logit_cosine)
        and control_gate_passed
    )
    report = {
        "passed": passed,
        "teacher_replay": str(args.teacher_replay.resolve()),
        "candidate": str(args.nano_vllm_path.resolve()),
        "diagnostic_config": diagnostic_config,
        "engine": {"enforce_eager": bool(args.enforce_eager)},
        "teacher_request_count": len(request_ids),
        "calls": len(comparisons),
        "token_agreement": token_agreement,
        "required_token_agreement": args.require_token_agreement,
        "minimum_logit_cosine": min_cosine,
        "required_minimum_logit_cosine": args.min_logit_cosine,
        "first_token_divergence": first_divergence,
        "control_tokens": {
            "bos_token_id": args.bos_token_id,
            "pad_token_id": args.pad_token_id,
            "margin": "BOS logit minus PAD logit",
            "comparisons": len(control_margin_pairs),
            "missing_allowed": bool(args.allow_missing_control_margins),
            "order_agreement": control_order_agreement,
            "required_order_agreement": args.require_bos_pad_order_agreement,
            "max_absolute_margin_delta": max_control_margin_delta,
            "allowed_max_absolute_margin_delta": args.max_bos_pad_margin_delta,
            "minimum_teacher_margin": min((left for left, _ in control_margin_pairs), default=None),
            "minimum_candidate_margin": min(
                (right for _, right in control_margin_pairs), default=None
            ),
            "maximum_teacher_margin": max((left for left, _ in control_margin_pairs), default=None),
            "maximum_candidate_margin": max(
                (right for _, right in control_margin_pairs), default=None
            ),
            "first_order_divergence": first_control_order_divergence,
        },
        "mean_call_ms": 1000.0 * sum(call_times) / len(call_times),
        "comparisons": comparisons,
    }
    if args.attribution_mode != "off":
        attribution = {
            "profiling_only": True,
            "writes_model_artifact": False,
            "mode": args.attribution_mode,
            "layers": sorted(attribution_layers),
            "windows": sorted(attribution_windows),
            "worker_install": attribution_worker_result,
            "worker_finalize": attribution_finalize_result,
            "target_gate": target_window_summary(
                comparisons,
                attribution_windows,
                max_margin_delta=args.max_bos_pad_margin_delta,
            ),
        }
        if args.attribution_reference_dir:
            attribution["layer_comparison"] = compare_snapshot_directories(
                args.attribution_reference_dir, attribution_output
            )
        report["attribution"] = attribution
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
