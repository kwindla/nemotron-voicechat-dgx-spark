#!/usr/bin/env python3
"""Fixed-token EarTTS-only correctness gate with exact codec and external ASR."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path

PUBLIC_REVISION = "fb0f94eaf4d03ddc430f39565229393fa1b50c26"
PUBLIC_SHA256 = "d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b"
REFERENCE_TEXT = "The answer is five."
# Captured from the passing public Nano-vLLM + native-EarTTS teacher session
# e4cde111-0dbd-4a68-b74a-ecdb7112126a. Frames 0--53 are required PAD context;
# the delivered assistant response occupies model frames 54--134.
PREFIX_PAD_FRAMES = 54
RESPONSE_TEXT_TOKENS = [1, 1784, 4832, 1395, 6184, 1046] + [12] * 74 + [2]
TEACHER_TEXT_TOKENS = [12] * PREFIX_PAD_FRAMES + RESPONSE_TEXT_TOKENS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_pcm16_wav(path: Path, waveform: object, sample_rate: int) -> None:
    import numpy as np

    values = waveform.detach().cpu().float().numpy()
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def decode_streaming_codec(
    audio_codec: object,
    codes: object,
    *,
    mode: str,
    frames_per_chunk: int,
    samples_per_frame: int,
    history_size: int,
    silence_tokens: object | None = None,
    codec_cache: object | None = None,
    replace_control_codes: object | None = None,
) -> tuple[object, list[int]]:
    """Replay acoustic codes through the wrapper's production decode path.

    ``cache`` mirrors ``NemotronVoicechatInferenceWrapper.infer_one_step``
    when ``use_codec_cache=true``: only newly generated codes are decoded and
    one cache object persists across every chunk in the stream.

    ``history`` mirrors its bounded fallback: a silence-initialized
    ``audio_toks_buffer`` is shifted once per new frame, the complete bounded
    history is decoded once per chunk, and only the newest output frames are
    delivered.  The helper deliberately does not flush at response EOS because
    the production wrapper preserves codec state for the entire conversation.
    """
    import torch

    if mode not in {"cache", "history"}:
        raise ValueError(f"unsupported codec streaming mode: {mode}")
    if codes.ndim != 3 or codes.shape[0] != 1:
        raise ValueError(f"expected acoustic codes shaped [1, T, Q], got {tuple(codes.shape)}")
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be positive")
    if samples_per_frame <= 0:
        raise ValueError("samples_per_frame must be positive")
    if mode == "cache" and codec_cache is None:
        raise ValueError("cache mode requires a codec_cache")
    if mode == "history":
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if frames_per_chunk > history_size:
            raise ValueError("frames_per_chunk cannot exceed history_size")
        if silence_tokens is None:
            raise ValueError("history mode requires silence_tokens")
        audio_toks_buffer = silence_tokens.view(1, 1, -1).expand(-1, history_size, -1).clone()
    else:
        audio_toks_buffer = None

    decoded_chunks: list[object] = []
    decoded_samples: list[int] = []
    for start in range(0, codes.shape[1], frames_per_chunk):
        new_codes = codes[:, start : start + frames_per_chunk]
        chunk_frames = int(new_codes.shape[1])
        code_len = torch.tensor([chunk_frames], dtype=torch.long, device=new_codes.device)
        if mode == "cache":
            decode_codes = (
                replace_control_codes(new_codes) if replace_control_codes is not None else new_codes
            )
            decoded, _ = audio_codec.decode(decode_codes, code_len, cache=codec_cache)
        else:
            # The wrapper shifts once inside each autoregressive frame, before
            # decoding the completed multi-frame chunk.
            for frame_offset in range(chunk_frames):
                audio_toks_buffer = torch.cat(
                    [
                        audio_toks_buffer[:, 1:],
                        new_codes[:, frame_offset : frame_offset + 1],
                    ],
                    dim=1,
                )
            history_len = torch.tensor([history_size], dtype=torch.long, device=new_codes.device)
            decoded_history, _ = audio_codec.decode(audio_toks_buffer, history_len)
            wanted_samples = samples_per_frame * chunk_frames
            if decoded_history.shape[-1] < wanted_samples:
                raise RuntimeError(
                    "codec history decode returned fewer samples than the "
                    f"wrapper tail slice requires: {decoded_history.shape[-1]} "
                    f"< {wanted_samples}"
                )
            decoded = decoded_history[:, :, -wanted_samples:]
        decoded_chunks.append(decoded)
        decoded_samples.append(int(decoded.shape[-1]))

    if not decoded_chunks:
        raise ValueError("cannot stream-decode an empty acoustic-code sequence")
    return torch.cat(decoded_chunks, dim=-1), decoded_samples


def response_audio_interval(
    waveform: object, *, prefix_frames: int, samples_per_frame: int
) -> object:
    """Remove fixture prefix frames at the same 80 ms output boundary."""
    start_sample = prefix_frames * samples_per_frame
    if waveform.shape[-1] <= start_sample:
        raise RuntimeError(f"decoded waveform has no response interval after sample {start_sample}")
    return waveform[..., start_sample:]


def load_live_acoustic_capture(
    capture_dir: Path,
    *,
    request_id: str | None = None,
    stream_id: int | None = None,
    turn_index: int = 0,
    post_eos_frames: int = 0,
    allow_incomplete: bool = False,
) -> dict[str, object]:
    """Load and verify one live trajectory for production cached-codec replay."""
    import torch

    capture_dir = capture_dir.resolve()
    metadata_path = capture_dir / "metadata.json"
    if not metadata_path.is_file():
        candidates = sorted(capture_dir.glob("*/metadata.json"))
        if len(candidates) != 1:
            raise ValueError(
                "capture path must be one run directory, or contain exactly one run; "
                f"found {len(candidates)} metadata files"
            )
        metadata_path = candidates[0]
        capture_dir = metadata_path.parent
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("schema") != "nemotron-live-eartts-acoustic-capture-v1":
        raise ValueError(f"unsupported capture metadata schema: {metadata.get('schema')}")
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if not metadata.get("source_hashes") or not metadata.get("model_config_sha256"):
        raise ValueError("capture metadata lacks exact source/config hashes")

    records = []
    for frame_path in sorted(capture_dir.glob("frame-*.pt")):
        record = torch.load(frame_path, map_location="cpu", weights_only=False)
        if record.get("schema") != "nemotron-live-eartts-acoustic-frame-v1":
            raise ValueError(f"unsupported frame schema in {frame_path.name}")
        if record.get("capture_metadata_sha256") != metadata_sha256:
            raise ValueError(f"metadata hash mismatch in {frame_path.name}")
        if request_id is not None and record.get("request_id") != request_id:
            continue
        if stream_id is not None and record.get("stream_id") != stream_id:
            continue
        records.append(record)
    if not records:
        raise ValueError("no captured acoustic frames matched the requested trajectory")
    records.sort(key=lambda record: int(record["sequence"]))
    sequences = [int(record["sequence"]) for record in records]
    if sequences != list(range(sequences[0], sequences[-1] + 1)):
        raise ValueError("captured trajectory contains a sequence gap")

    trajectory_keys = {(record.get("request_id"), record.get("stream_id")) for record in records}
    if len(trajectory_keys) != 1:
        raise ValueError(
            "capture contains multiple request/stream trajectories; select one with "
            "--captured-request-id and --captured-stream-id"
        )
    bos_indices = [
        index for index, record in enumerate(records) if record["text_control"]["is_bos"]
    ]
    if turn_index < 0 or turn_index >= len(bos_indices):
        raise ValueError(
            f"turn_index {turn_index} is unavailable; capture has {len(bos_indices)} BOS events"
        )
    response_start = bos_indices[turn_index]
    response_end = next(
        (
            index
            for index in range(response_start, len(records))
            if records[index]["text_control"]["is_eos"]
        ),
        None,
    )
    incomplete = response_end is None
    if incomplete and not allow_incomplete:
        raise ValueError(f"captured turn {turn_index} has no EOS event")
    if incomplete:
        response_end = len(records) - 1
    if post_eos_frames < 0:
        raise ValueError("post_eos_frames cannot be negative")
    replay_end = min(response_end + post_eos_frames, len(records) - 1)
    records = records[: replay_end + 1]

    tensors = {}
    for key in ("raw_generated_code", "recurrent_code", "decoder_code"):
        values = [record[key] for record in records]
        if any(tuple(value.shape[:2]) != (1, 1) for value in values):
            raise ValueError(f"captured {key} does not contain one frame per record")
        tensors[key] = torch.cat(values, dim=1)
    text_tokens = torch.cat([record["current_subword_id"] for record in records], dim=1)
    return {
        "capture_dir": capture_dir,
        "metadata_path": metadata_path,
        "metadata_sha256": metadata_sha256,
        "metadata": metadata,
        "request_id": records[0].get("request_id"),
        "stream_id": records[0].get("stream_id"),
        "turn_index": turn_index,
        "response_start_frame": response_start,
        "response_end_frame": response_end,
        "replay_end_frame": replay_end,
        "post_eos_frames": replay_end - response_end,
        "incomplete": incomplete,
        "frame_count": len(records),
        "text_tokens": text_tokens,
        **tensors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--nano-skeleton", type=Path, required=True)
    parser.add_argument("--eartts-vllm-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("native", "vllm"),
        default="vllm",
        help="EarTTS implementation to exercise with the identical fixture",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--max-wer", type=float, default=0.5)
    parser.add_argument(
        "--codec-streaming-mode",
        choices=("cache", "history"),
        default="cache",
        help=(
            "Production codec replay to Parakeet-gate. cache is the released "
            "streaming default; history exercises the bounded fallback."
        ),
    )
    parser.add_argument(
        "--codec-frames-per-chunk",
        type=int,
        default=1,
        help="Number of 80 ms acoustic frames decoded per production chunk",
    )
    parser.add_argument(
        "--codec-token-history-size",
        type=int,
        default=30,
        help="Bounded fallback history; ignored in cache mode",
    )
    parser.add_argument("--debug-tensors", action="store_true")
    parser.add_argument(
        "--teacher-acoustic-codes",
        type=Path,
        help="Feed these prior-frame codes while retaining backend predictions",
    )
    parser.add_argument(
        "--captured-live-trajectory",
        type=Path,
        help=(
            "Skip EarTTS generation and replay the exact decoder_code frames from "
            "an opt-in live capture through the production cached codec"
        ),
    )
    parser.add_argument("--captured-request-id")
    parser.add_argument("--captured-stream-id", type=int)
    parser.add_argument("--captured-turn-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug"
    if args.debug_tensors:
        debug_dir.mkdir(parents=True, exist_ok=True)
        os.environ["EARTTS_DEBUG_DIR"] = str(debug_dir)
        os.environ["EARTTS_DEBUG_MAX_CALLS"] = "40"
    checkpoint_path = args.checkpoint_root / "model.safetensors"
    if not args.skip_checkpoint_hash and sha256(checkpoint_path) != PUBLIC_SHA256:
        raise SystemExit("exact public checkpoint SHA256 mismatch")

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NVIDIA_TF32_OVERRIDE": "1",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        }
    )
    sys.path.insert(0, str(args.speech_root.resolve()))

    import torch
    import torchaudio
    from nemo.collections.speechlm2.inference.model_wrappers.model_factory import (
        create_model,
    )
    from nemo.collections.speechlm2.models.duplex_ear_tts import (
        DuplexEARTTS,
        replace_control_speech_codes,
    )
    from safetensors import safe_open

    torch.manual_seed(0)
    config = json.loads((args.checkpoint_root / "config.json").read_text())
    tts_config = config["model"]["speech_generation"]
    tts_config["model"]["pretrained_lm_name"] = str(args.nano_skeleton.resolve())
    tts_config["model"]["tts_config"]["cas_config"]["pretrained_tokenizer_name"] = str(
        args.nano_skeleton.resolve()
    )
    tts_config["exp_manager"]["explicit_log_dir"] = str(args.output_dir)

    started = time.perf_counter()
    duplex = DuplexEARTTS(tts_config).eval()
    selected: dict[str, torch.Tensor] = {}
    component_prefixes = (
        "tts_model.audio_codec.",
        "tts_model.audio_prompt_latents.",
        "tts_model.codec_silence_tokens",
        "tts_model._control_codes",
    )
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as source:
        for key in source.keys():
            # The public S2S checkpoint wraps DuplexEARTTS under one outer
            # ``tts_model``. Native comparison needs the complete component;
            # vLLM comparison retains only its codec and prompt shell.
            if key.startswith("tts_model.") and (
                args.backend == "native" or key.startswith(component_prefixes)
            ):
                selected[key.removeprefix("tts_model.")] = source.get_tensor(key)
    duplex.maybe_recreate_cached_audio_prompt_latents_structure(selected)
    incompatible = duplex.load_state_dict(selected, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected selected checkpoint tensors: {unexpected}")
    codec_state_names = set(duplex.audio_codec.state_dict())
    loaded_codec_names = {
        name.removeprefix("audio_codec.") for name in selected if name.startswith("audio_codec.")
    }
    missing_codec = sorted(codec_state_names - loaded_codec_names)
    if missing_codec:
        raise RuntimeError(f"public codec tensors missing: {missing_codec[:10]}")
    if "Aria" not in duplex.audio_prompt_latents:
        raise RuntimeError("public Aria prompt latent was not loaded")

    vllm_tts = None
    if args.backend == "vllm":
        # Free native EarTTS before starting the exact public vLLM engine,
        # while retaining only the codec and prompt-preparation shell.
        del duplex.tts_model
        gc.collect()
        duplex = duplex.to("cuda").eval()
        torch.cuda.empty_cache()
        vllm_tts = create_model(
            model=str(args.checkpoint_root.resolve()),
            engine_type="vllm_eartts",
            vllm_config={
                "model_path": str(args.eartts_vllm_path.resolve()),
                "engine_path": str(args.eartts_vllm_path.resolve()),
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.18,
                "dtype": "float32",
                "skip_tokenizer_init": True,
                "enforce_eager": bool(args.enforce_eager),
            },
        )
        object.__setattr__(duplex, "tts_model", vllm_tts)
    else:
        duplex = duplex.to("cuda").eval()
        torch.cuda.empty_cache()

    if args.debug_tensors and args.backend == "native":
        # Capture the native tensors corresponding to the ephemeral trace in
        # instrument_vllm_eartts.py. Hooks leave inference arithmetic intact.
        import nemo.collections.speechlm2.modules.ear_tts_model as native_eartts

        mog_head = duplex.tts_model.mog_head
        original_infer = mog_head.infer
        original_gumbel_like = native_eartts.gumbel_like
        debug_call_index = 0
        debug_forward_index = 0

        def capture_native_backbone(_module, _args, kwargs, output):
            nonlocal debug_forward_index
            if debug_forward_index < 8:
                past = kwargs.get("past_key_values")
                cache_length = None
                if past is not None and hasattr(past, "get_seq_length"):
                    cache_length = int(past.get_seq_length())
                torch.save(
                    {
                        "inputs_embeds": kwargs["inputs_embeds"].detach().cpu(),
                        "attention_mask": (
                            kwargs.get("attention_mask").detach().cpu()
                            if kwargs.get("attention_mask") is not None
                            else None
                        ),
                        "position_ids": (
                            kwargs.get("position_ids").detach().cpu()
                            if kwargs.get("position_ids") is not None
                            else None
                        ),
                        "cache_length_after_forward": cache_length,
                        "hidden_states": output.last_hidden_state.detach().cpu(),
                    },
                    debug_dir / f"native-forward-{debug_forward_index:03d}.pt",
                )
            debug_forward_index += 1

        duplex.tts_model.backbone.register_forward_hook(capture_native_backbone, with_kwargs=True)

        def traced_infer(x, guidance_scale=0.0, top_p_or_k=1.0):
            nonlocal debug_call_index
            captured = {}

            def capture_post_mlp(_module, _inputs, output):
                captured["post_mlp_pre_cfg"] = output.detach().cpu()

            def capture_logits(_module, inputs, output):
                captured["post_cfg"] = inputs[0].detach().cpu()
                captured["raw_logits"] = output.detach().cpu()

            def traced_gumbel(tensor, eps=1e-8):
                value = original_gumbel_like(tensor, eps)
                captured["gumbel"] = value.detach().cpu()
                return value

            mlp_hook = mog_head.mlp_stack.register_forward_hook(capture_post_mlp)
            logits_hook = mog_head.proj_logits.register_forward_hook(capture_logits)
            native_eartts.gumbel_like = traced_gumbel
            try:
                result = original_infer(x, guidance_scale, top_p_or_k)
            finally:
                native_eartts.gumbel_like = original_gumbel_like
                mlp_hook.remove()
                logits_hook.remove()

            raw_logits = captured["raw_logits"]
            if isinstance(top_p_or_k, float):
                from transformers.generation.logits_process import TopPLogitsWarper

                filtered_logits = TopPLogitsWarper(top_p_or_k)(
                    None, raw_logits.view(-1, raw_logits.shape[-1])
                ).view_as(raw_logits)
            else:
                from transformers.generation.logits_process import TopKLogitsWarper

                filtered_logits = TopKLogitsWarper(top_p_or_k)(
                    None, raw_logits.view(-1, raw_logits.shape[-1])
                ).view_as(raw_logits)
            mixture_indices = (
                torch.nn.functional.log_softmax(filtered_logits, dim=-1) + captured["gumbel"]
            ).argmax(-1)
            torch.save(
                {
                    "input": x.detach().cpu(),
                    **captured,
                    "filtered_logits": filtered_logits,
                    "mixture_indices": mixture_indices,
                    "mu_with_residual": result[0].detach().cpu(),
                    "logs": result[1].detach().cpu(),
                    "guidance_scale": guidance_scale,
                },
                debug_dir / f"native-mog-{debug_call_index:03d}.pt",
            )
            debug_call_index += 1
            return result

        mog_head.infer = traced_infer

    duplex.set_init_inputs(speaker_name="Aria")
    init_inputs = duplex.get_init_inputs(B=1)
    prompt_token_ids = init_inputs["subword_ids"].squeeze().cpu().tolist()
    init_inputs.update(
        {
            "use_cache": True,
            "past_key_values": None,
            "guidance_enabled": True,
        }
    )
    prefill_started = time.perf_counter()
    if args.backend == "vllm":
        outputs = duplex.tts_model(
            init_inputs,
            request_id="component-eartts",
            prompt_token_ids=prompt_token_ids,
        )
    else:
        outputs = duplex.tts_model(**init_inputs)
    prefill_seconds = time.perf_counter() - prefill_started
    code = init_inputs["code"][:, -1:].clone()
    previous_token = init_inputs["subword_ids"][:, -1:].clone()
    past_key_values = outputs.past_key_values
    generated: list[torch.Tensor] = []
    step_times: list[float] = []
    generation_config = duplex._get_generation_config(guidance_enabled=True)
    teacher_codes = None
    if args.teacher_acoustic_codes:
        teacher_codes = torch.load(args.teacher_acoustic_codes, map_location="cuda")
        if tuple(teacher_codes.shape) != (1, len(TEACHER_TEXT_TOKENS), 31):
            raise RuntimeError(f"teacher acoustic shape mismatch: {tuple(teacher_codes.shape)}")

    for frame_index, token_id in enumerate(TEACHER_TEXT_TOKENS):
        current_token = torch.tensor([[token_id]], device="cuda", dtype=torch.long)
        step_started = time.perf_counter()
        code, past_key_values = duplex.infer_codes_one_step(
            current_subword_id=current_token,
            prev_subword_id=previous_token,
            current_subword_mask=torch.ones_like(current_token, dtype=torch.bool),
            prev_audio_tokens=code,
            past_key_values=past_key_values,
            guidance_enabled=True,
            generation_config=generation_config,
            ignore_eos_flag_stop=True,
            request_id="component-eartts" if args.backend == "vllm" else None,
        )
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_started)
        generated.append(code.detach().clone())
        if teacher_codes is not None:
            code = teacher_codes[:, frame_index : frame_index + 1]
        previous_token = current_token

    codes = torch.cat(generated, dim=1)
    expected_shape = (1, len(TEACHER_TEXT_TOKENS), 31)
    if tuple(codes.shape) != expected_shape:
        raise RuntimeError(f"code ordering mismatch: {tuple(codes.shape)} != {expected_shape}")
    code_min, code_max = int(codes.min()), int(codes.max())
    if code_min < 0 or code_max >= 1024:
        raise RuntimeError(f"acoustic code range is invalid: [{code_min}, {code_max}]")

    code_path = args.output_dir / "acoustic-codes.pt"
    torch.save(codes.cpu(), code_path)
    # Keep the original whole-history decode as a diagnostic.  It is useful
    # for separating acoustic-code correctness from the stateful codec replay,
    # but it is not what the live server sends over the wire.
    lengths = torch.tensor([codes.shape[1]], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        offline_waveform, offline_waveform_lengths = duplex.audio_codec.decode(
            codes.long(), lengths
        )
    offline_waveform = offline_waveform[:, :, : int(offline_waveform_lengths[0])]
    offline_waveform = (
        response_audio_interval(
            offline_waveform,
            prefix_frames=PREFIX_PAD_FRAMES,
            samples_per_frame=duplex.target_samples_per_frame,
        )[0, 0]
        .detach()
        .cpu()
    )
    offline_wav_path = args.output_dir / "eartts-component-offline.wav"
    save_pcm16_wav(offline_wav_path, offline_waveform, duplex.target_sample_rate)

    # Required gate: reconstruct the waveform exactly as the streaming wrapper
    # does.  The released path uses one persistent causal codec cache and never
    # flushes it at assistant EOS; the bounded history path remains available
    # to exercise the wrapper's non-cache fallback and last-frame slicing.
    codec_cache = None
    control_replacer = None
    if args.codec_streaming_mode == "cache":
        from nemo.collections.speechlm2.modules.ear_tts_vae_codec import (
            CausalConv1dCache,
        )

        codec_cache = CausalConv1dCache()
        if hasattr(duplex, "_control_codes"):
            control_codes = duplex._control_codes
            codec_silence_tokens = getattr(duplex, "codec_silence_tokens", None)

            def control_replacer(value):
                return replace_control_speech_codes(
                    value,
                    control_codes,
                    codec_silence_tokens,
                )

    with torch.inference_mode():
        streaming_waveform, streaming_chunk_samples = decode_streaming_codec(
            duplex.audio_codec,
            codes.long(),
            mode=args.codec_streaming_mode,
            frames_per_chunk=args.codec_frames_per_chunk,
            samples_per_frame=duplex.target_samples_per_frame,
            history_size=args.codec_token_history_size,
            silence_tokens=duplex.codec_silence_tokens,
            codec_cache=codec_cache,
            replace_control_codes=control_replacer,
        )
    streaming_waveform = (
        response_audio_interval(
            streaming_waveform,
            prefix_frames=PREFIX_PAD_FRAMES,
            samples_per_frame=duplex.target_samples_per_frame,
        )[0, 0]
        .detach()
        .cpu()
    )
    wav_path = args.output_dir / "eartts-component.wav"
    save_pcm16_wav(wav_path, streaming_waveform, duplex.target_sample_rate)

    # Release the vLLM engine and codec before loading the independent ASR.
    if args.backend == "vllm":
        duplex.tts_model.abort_request("component-eartts")
        duplex.tts_model.shutdown()
    del duplex
    if vllm_tts is not None:
        del vllm_tts
    gc.collect()
    torch.cuda.empty_cache()

    resampled = torchaudio.functional.resample(streaming_waveform, 22050, 16000)
    wav_16k_path = args.output_dir / "eartts-component-16k.wav"
    save_pcm16_wav(wav_16k_path, resampled, 16000)
    offline_resampled = torchaudio.functional.resample(offline_waveform, 22050, 16000)
    offline_wav_16k_path = args.output_dir / "eartts-component-offline-16k.wav"
    save_pcm16_wav(offline_wav_16k_path, offline_resampled, 16000)
    report = {
        "passed": False,
        "pre_asr_passed": True,
        "pending_external_asr": True,
        "checkpoint": {
            "repository": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
            "revision": PUBLIC_REVISION,
            "sha256": PUBLIC_SHA256,
            "hash_verified_this_run": not args.skip_checkpoint_hash,
        },
        "engine": {
            "backend": args.backend,
            "eartts_path": str(args.eartts_vllm_path.resolve()),
            "enforce_eager": bool(args.enforce_eager) if args.backend == "vllm" else None,
            "guidance_scale": 0.2,
            "top_p_or_k": 0.95,
        },
        "fixture": {
            "source_session": "e4cde111-0dbd-4a68-b74a-ecdb7112126a",
            "reference_text": REFERENCE_TEXT,
            "token_count": len(TEACHER_TEXT_TOKENS),
            "prefix_pad_frames": PREFIX_PAD_FRAMES,
            "response_token_count": len(RESPONSE_TEXT_TOKENS),
            "tokens": TEACHER_TEXT_TOKENS,
            "teacher_acoustic_codes": (
                str(args.teacher_acoustic_codes.resolve()) if args.teacher_acoustic_codes else None
            ),
        },
        "codes": {
            "path": str(code_path),
            "shape": list(codes.shape),
            "dtype": str(codes.dtype),
            "min": code_min,
            "max": code_max,
        },
        "audio": {
            "path": str(wav_path),
            "construction": "production_streaming_codec",
            "streaming_mode": args.codec_streaming_mode,
            "frames_per_chunk": args.codec_frames_per_chunk,
            "codec_token_history_size": (
                args.codec_token_history_size if args.codec_streaming_mode == "history" else None
            ),
            "codec_cache_persisted_across_chunks": (args.codec_streaming_mode == "cache"),
            "codec_cache_flushed_at_eos": False,
            "chunk_count": len(streaming_chunk_samples),
            "chunk_samples": streaming_chunk_samples,
            "sample_rate": 22050,
            "samples": int(streaming_waveform.numel()),
            "duration_seconds": streaming_waveform.numel() / 22050,
            "peak": float(streaming_waveform.abs().max()),
            "rms": float(streaming_waveform.square().mean().sqrt()),
        },
        "external_asr": {
            "status": "pending",
            "max_wer": args.max_wer,
            "is_required_gate": True,
        },
        "asr_inputs": {
            "streaming": {
                "path": wav_16k_path.name,
                "bytes": wav_16k_path.stat().st_size,
                "sha256": sha256(wav_16k_path),
            },
            "offline": {
                "path": offline_wav_16k_path.name,
                "bytes": offline_wav_16k_path.stat().st_size,
                "sha256": sha256(offline_wav_16k_path),
            },
        },
        "offline_diagnostic": {
            "path": str(offline_wav_path),
            "construction": "single_whole_history_decode",
            "sample_rate": 22050,
            "samples": int(offline_waveform.numel()),
            "duration_seconds": offline_waveform.numel() / 22050,
            "peak": float(offline_waveform.abs().max()),
            "rms": float(offline_waveform.square().mean().sqrt()),
            "external_asr": {
                "status": "pending",
                "is_required_gate": False,
            },
        },
        "timing": {
            "startup_seconds": prefill_started - started,
            "prefill_seconds": prefill_seconds,
            "decode_step_mean_ms": 1000 * sum(step_times) / len(step_times),
            "decode_step_max_ms": 1000 * max(step_times),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
