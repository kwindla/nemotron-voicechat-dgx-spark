#!/usr/bin/env python3
"""Isolated CPU process for incremental VoiceChat codec decode."""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing.connection import Client
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--pointwise-linear", action="store_true")
    return parser.parse_args()


def load_codec_state(checkpoint_root: Path, *, safe_open):
    """Load the codec tensors from the public combined Voicechat checkpoint."""
    combined = checkpoint_root / "model.safetensors"
    if not combined.is_file():
        raise FileNotFoundError(f"Missing combined model.safetensors in {checkpoint_root}")
    prefix = "tts_model.audio_codec."
    state = {}
    with safe_open(str(combined), framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith(prefix):
                state[name.removeprefix(prefix)] = source.get_tensor(name)
    if not state:
        raise RuntimeError(f"No {prefix} tensors found in {combined}")
    return state, {
        "layout": "public_combined",
        "path": str(combined),
        "source_prefix": prefix,
        "tensor_count": len(state),
    }


def main() -> None:
    args = parse_args()
    cpus = tuple(int(value) for value in args.cpus.split(","))
    os.sched_setaffinity(0, cpus)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["NVPL_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    os.environ["GOMP_CPU_AFFINITY"] = " ".join(str(cpu) for cpu in cpus)

    import torch
    from cpu_codec_offload import SlicedDepthwiseConv1d, replace_pointwise_convs
    from nemo.collections.speechlm2.modules.ear_tts_vae_codec import (
        CausalConv1dCache,
        RVQVAEModel,
    )
    from omegaconf import DictConfig
    from safetensors import safe_open

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint_root = Path(args.checkpoint_root)
    config = json.loads((checkpoint_root / "config.json").read_text())
    codec_config = config["model"]["speech_generation"]["model"]["codec_config"]
    model = RVQVAEModel(DictConfig(codec_config)).eval()
    codec_state, checkpoint_details = load_codec_state(checkpoint_root, safe_open=safe_open)
    model.load_state_dict(codec_state, strict=True)
    replacements = 0
    for module in model.modules():
        candidate = getattr(module, "dwconv", None)
        if (
            isinstance(candidate, torch.nn.Conv1d)
            and candidate.groups == candidate.in_channels
            and candidate.out_channels == candidate.in_channels
        ):
            module.dwconv = SlicedDepthwiseConv1d.create(torch, candidate)
            replacements += 1
    if replacements != 18:
        raise RuntimeError(f"expected 18 depthwise replacements, got {replacements}")
    pointwise_replacements = 0
    if args.pointwise_linear:
        pointwise_replacements = replace_pointwise_convs(torch, model)
        if pointwise_replacements != 38:
            raise RuntimeError(f"expected 38 pointwise replacements, got {pointwise_replacements}")
    cache = CausalConv1dCache()
    connection = Client(args.socket, family="AF_UNIX")
    connection.send(
        {
            "status": "ready",
            "pid": os.getpid(),
            "threads": args.threads,
            "cpus": list(cpus),
            "depthwise_replacements": replacements,
            "pointwise_replacements": pointwise_replacements,
            "checkpoint": checkpoint_details,
        }
    )
    while True:
        request = connection.recv()
        if request["op"] == "shutdown":
            break
        if request["op"] not in {"decode", "warmup"}:
            connection.send({"status": "error", "message": "unknown operation"})
            continue
        try:
            is_warmup = request["op"] == "warmup"
            if is_warmup or request.get("reset_cache"):
                cache = CausalConv1dCache()
            codes = torch.from_numpy(request["codes"])
            lengths = torch.from_numpy(request["lengths"])
            start = time.perf_counter()
            with torch.inference_mode():
                audio, output_lengths = model.decode(codes, lengths, cache=cache)
            worker_ms = (time.perf_counter() - start) * 1000
            if is_warmup:
                # No session may inherit warmup history.
                cache = CausalConv1dCache()
                connection.send({"status": "ok", "worker_ms": worker_ms})
                continue
            connection.send(
                {
                    "status": "ok",
                    "sequence": request["sequence"],
                    "audio": audio.numpy(),
                    "lengths": output_lengths.numpy(),
                    "worker_ms": worker_ms,
                }
            )
        except Exception as error:
            connection.send(
                {
                    "status": "error",
                    "sequence": request.get("sequence"),
                    "message": repr(error),
                }
            )
    connection.close()


if __name__ == "__main__":
    main()
