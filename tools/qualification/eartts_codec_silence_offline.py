#!/usr/bin/env python3
"""Offline replay of captured EarTTS codes through codec implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import wave
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dbfs(torch: Any, value: Any) -> float:
    rms = float(torch.sqrt(torch.mean(value.float() ** 2)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _save_wav(path: Path, torch: Any, frames: list[Any], sample_rate: int) -> None:
    import numpy as np

    waveform = torch.cat([frame.reshape(-1) for frame in frames]).float().numpy()
    pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _load_codec(args: argparse.Namespace):
    import torch
    from nemo.collections.speechlm2.modules.ear_tts_vae_codec import RVQVAEModel
    from omegaconf import DictConfig
    from safetensors import safe_open

    from nemotron_voicechat_runtime.cpu_codec_offload import (
        SlicedDepthwiseConv1d,
        replace_pointwise_convs,
    )

    config = json.loads((args.checkpoint_root / "config.json").read_text())
    codec_config = config["model"]["speech_generation"]["model"]["codec_config"]
    codec = RVQVAEModel(DictConfig(codec_config)).eval()
    prefix = "tts_model.audio_codec."
    state = {}
    checkpoint = args.checkpoint_root / "model.safetensors"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith(prefix):
                state[name.removeprefix(prefix)] = source.get_tensor(name)
    codec.load_state_dict(state, strict=True)
    codec.to(device="cpu", dtype=torch.float32)

    depthwise_replacements = 0
    pointwise_replacements = 0
    if args.variant != "reference":
        for module in codec.modules():
            candidate = getattr(module, "dwconv", None)
            if (
                isinstance(candidate, torch.nn.Conv1d)
                and candidate.groups == candidate.in_channels
                and candidate.out_channels == candidate.in_channels
            ):
                module.dwconv = SlicedDepthwiseConv1d.create(torch, candidate)
                depthwise_replacements += 1
        if depthwise_replacements != 18:
            raise RuntimeError(
                f"expected 18 depthwise replacements, got {depthwise_replacements}"
            )
    if args.variant in {"production", "persistent_off", "fresh_response_cache"}:
        pointwise_replacements = replace_pointwise_convs(torch, codec)
        if pointwise_replacements != 38:
            raise RuntimeError(
                f"expected 38 pointwise replacements, got {pointwise_replacements}"
            )
    return codec, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "tensor_count": len(state),
        "depthwise_replacements": depthwise_replacements,
        "pointwise_replacements": pointwise_replacements,
        "dtype": "float32",
        "device": "cpu",
    }


def _decode(
    *,
    torch: Any,
    codec: Any,
    codes: list[Any],
    response_start: int,
    fresh_response_cache: bool,
) -> tuple[list[Any], list[float]]:
    from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache

    cache = CausalConv1dCache()
    frames = []
    elapsed_ms = []
    with torch.inference_mode():
        for frame_index, code in enumerate(codes):
            if fresh_response_cache and frame_index == response_start:
                cache = CausalConv1dCache()
            started = time.perf_counter()
            audio, lengths = codec.decode(
                code.long(), torch.ones((1,), dtype=torch.long), cache=cache
            )
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)
            length = int(lengths.reshape(-1)[0])
            frames.append(audio[..., :length].detach().cpu().reshape(-1))
    return frames, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--response", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "production",
            "pointwise_off",
            "persistent_off",
            "fresh_response_cache",
            "reference",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", default="16,17")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    cpus = tuple(int(value) for value in args.cpus.split(","))
    os.sched_setaffinity(0, cpus)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["NVPL_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    os.environ["GOMP_CPU_AFFINITY"] = " ".join(str(cpu) for cpu in cpus)

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    analysis = json.loads(args.analysis.read_text())
    target = next(
        row
        for row in analysis["responses"]
        if row["attempt"] == args.attempt and row["response"] == args.response
    )
    request_id = str(args.attempt + 1)  # capture request 1 is startup warmup
    run_dirs = [path for path in args.capture_root.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected one capture run, found {len(run_dirs)}")
    records = {}
    for frame_path in sorted(run_dirs[0].glob("frame-*.pt")):
        record = torch.load(frame_path, map_location="cpu", weights_only=False)
        if str(record["request_id"]) == request_id:
            records[int(record["frame_index"])] = record
    end_frame = int(target["end_frame"])
    if sorted(records)[: end_frame + 1] != list(range(end_frame + 1)):
        raise RuntimeError("capture does not contain contiguous session history")
    response_start = int(target["start_frame"])
    response_end = end_frame
    decoder_codes = [records[index]["decoder_code"] for index in range(end_frame + 1)]
    raw_codes = [
        (
            records[index]["raw_generated_code"]
            if response_start <= index <= response_end
            else records[index]["decoder_code"]
        )
        for index in range(end_frame + 1)
    ]

    codec, codec_details = _load_codec(args)
    fresh_response_cache = args.variant == "fresh_response_cache"
    actual_frames = None
    actual_ms = None
    if args.variant == "production":
        actual_frames, actual_ms = _decode(
            torch=torch,
            codec=codec,
            codes=decoder_codes,
            response_start=response_start,
            fresh_response_cache=False,
        )
    raw_frames, raw_ms = _decode(
        torch=torch,
        codec=codec,
        codes=raw_codes,
        response_start=response_start,
        fresh_response_cache=fresh_response_cache,
    )

    captured = [records[index]["codec_pcm"].reshape(-1) for index in range(end_frame + 1)]
    positions = []
    for index in range(response_start, response_end + 1):
        raw_audio = raw_frames[index]
        captured_audio = captured[index]
        row = {
            "frame": index,
            "raw_dbfs": _dbfs(torch, raw_audio),
            "captured_dbfs": _dbfs(torch, captured_audio),
            "raw_equals_decoder_code": torch.equal(
                records[index]["raw_generated_code"], records[index]["decoder_code"]
            ),
            "policy_applied": bool(records[index]["decoder_pad_policy_applied"]),
            "tail_done": bool(records[index]["tail_done"]),
            "current_subword_id": int(records[index]["current_subword_id"]),
            "decode_ms": raw_ms[index],
        }
        if actual_frames is not None:
            actual = actual_frames[index]
            difference = actual - captured_audio
            row.update(
                {
                    "actual_decoder_dbfs": _dbfs(torch, actual),
                    "actual_vs_captured_max_abs": float(difference.abs().max()),
                    "actual_vs_captured_mean_abs": float(difference.abs().mean()),
                }
            )
        positions.append(row)

    raw_response_frames = raw_frames[response_start : response_end + 1]
    _save_wav(args.output / "raw-response.wav", torch, raw_response_frames, 22050)
    if actual_frames is not None:
        _save_wav(
            args.output / "actual-decoder-response.wav",
            torch,
            actual_frames[response_start : response_end + 1],
            22050,
        )
    with (args.output / "positions.jsonl").open("w", encoding="utf-8") as handle:
        for row in positions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    captured_silence_start = int(target["final_pcm_silence_start_frame"])
    raw_silent = [row["frame"] for row in positions if row["raw_dbfs"] <= -90.0]
    audible_after_captured_silence = [
        row["frame"]
        for row in positions
        if row["frame"] >= captured_silence_start and row["raw_dbfs"] > -90.0
    ]
    summary = {
        "schema": "eartts-codec-silence-offline-v1",
        "variant": args.variant,
        "variant_semantics": {
            "pointwise_linear": args.variant
            in {"production", "persistent_off", "fresh_response_cache"},
            "sliced_depthwise": args.variant != "reference",
            "cache_reset_at_response_start": fresh_response_cache,
            "persistent_off_note": (
                "EA_CPU_CODEC_PERSISTENT=0 changes worker lifetime at session finalization; "
                "the causal cache still persists across calls within the session."
                if args.variant == "persistent_off"
                else None
            ),
        },
        "codec": codec_details,
        "cpus": list(cpus),
        "threads": args.threads,
        "capture": {
            "root": str(args.capture_root),
            "run": run_dirs[0].name,
            "request_id": request_id,
            "attempt": args.attempt,
            "response": args.response,
            "session_id": target["session_id"],
            "start_frame": response_start,
            "end_frame": response_end,
            "captured_final_silence_start_frame": captured_silence_start,
        },
        "raw_decode": {
            "response_positions": len(positions),
            "silent_positions_at_or_below_minus_90_dbfs": len(raw_silent),
            "first_silent_position": raw_silent[0] if raw_silent else None,
            "audible_positions_at_or_after_captured_silence_start": len(
                audible_after_captured_silence
            ),
            "first_audible_position_at_or_after_captured_silence_start": (
                audible_after_captured_silence[0]
                if audible_after_captured_silence
                else None
            ),
            "captured_silence_start_dbfs": next(
                row["raw_dbfs"]
                for row in positions
                if row["frame"] == captured_silence_start
            ),
            "decode_elapsed_seconds": sum(raw_ms) / 1000.0,
        },
        "faithfulness": None,
    }
    if actual_frames is not None:
        response_actual = positions
        summary["faithfulness"] = {
            "decoder_history_replayed": True,
            "max_abs_error": max(row["actual_vs_captured_max_abs"] for row in response_actual),
            "mean_of_position_mean_abs_error": sum(
                row["actual_vs_captured_mean_abs"] for row in response_actual
            )
            / len(response_actual),
            "max_dbfs_difference": max(
                abs(row["actual_decoder_dbfs"] - row["captured_dbfs"])
                for row in response_actual
            ),
            "decode_elapsed_seconds": sum(actual_ms or []) / 1000.0,
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
