"""Narrow SoundFile replacement for TorchAudio 2.10 load/save on ARM64.

TorchAudio 2.10 delegates these functions to TorchCodec, for which no Linux
ARM64 wheel is published. The NVIDIA offline helpers only use the basic WAV
load/save forms, so patch just those two functions and leave resampling and all
model operations untouched.
"""

from __future__ import annotations

import os
from typing import Any


def _load(
    uri: str | os.PathLike,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
    **_: Any,
):
    del format, buffer_size, backend
    if not normalize:
        raise ValueError("The SoundFile adapter supports normalize=True only")

    import soundfile as sf
    import torch

    with sf.SoundFile(uri, mode="r") as wav_file:
        if frame_offset < 0:
            raise ValueError("frame_offset must be non-negative")
        wav_file.seek(frame_offset)
        frames = -1 if num_frames is None or num_frames < 0 else num_frames
        data = wav_file.read(frames=frames, dtype="float32", always_2d=True)
        sample_rate = wav_file.samplerate

    tensor = torch.from_numpy(data.copy())
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, sample_rate


def _save(
    uri: str | os.PathLike,
    src,
    sample_rate: int,
    channels_first: bool = True,
    format: str | None = None,
    encoding: str | None = None,
    bits_per_sample: int | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
    compression: Any = None,
    **_: Any,
) -> None:
    del buffer_size, backend
    if compression is not None:
        raise ValueError("The SoundFile adapter does not support compression")
    if src.ndim == 1:
        src = src.unsqueeze(0 if channels_first else 1)
    if src.ndim != 2:
        raise ValueError("Expected a 1D or 2D audio tensor")

    import soundfile as sf
    import torch

    data = src.detach().to(device="cpu", dtype=torch.float32).numpy()
    if channels_first:
        data = data.T

    subtype = None
    if encoding not in (None, "PCM_S", "PCM_F"):
        raise ValueError(f"Unsupported encoding: {encoding}")
    if bits_per_sample is not None:
        if encoding == "PCM_F" and bits_per_sample == 32:
            subtype = "FLOAT"
        elif encoding in (None, "PCM_S") and bits_per_sample in (16, 24, 32):
            subtype = f"PCM_{bits_per_sample}"
        else:
            raise ValueError("Unsupported encoding/bits_per_sample combination")

    sf.write(uri, data, sample_rate, format=format, subtype=subtype)


def install_torchaudio_soundfile_io():
    """Patch and return the imported torchaudio module."""
    import torchaudio

    torchaudio.load = _load
    torchaudio.save = _save
    return torchaudio
