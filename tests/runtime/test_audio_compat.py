from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="runtime image supplies PyTorch")

from nemotron_voicechat_runtime.audio_compat import install_torchaudio_soundfile_io

torchaudio = install_torchaudio_soundfile_io()


@pytest.mark.parametrize("channels", [1, 2])
def test_wav_round_trip_preserves_layout_and_rate(tmp_path, channels):
    rate = 22050
    frames = 4096
    source = torch.linspace(-0.8, 0.8, frames).repeat(channels, 1)
    if channels == 2:
        source[1] = source[1].flip(0)
    path = tmp_path / f"{channels}ch.wav"

    torchaudio.save(path, source, rate, bits_per_sample=16)
    restored, restored_rate = torchaudio.load(path)

    assert restored_rate == rate
    assert restored.shape == (channels, frames)
    torch.testing.assert_close(restored, source, atol=4e-5, rtol=0)


def test_frame_window(tmp_path):
    path = tmp_path / "window.wav"
    source = torch.linspace(-0.5, 0.5, 1000).unsqueeze(0)
    torchaudio.save(path, source, 16000, bits_per_sample=16)
    restored, _ = torchaudio.load(path, frame_offset=100, num_frames=200)
    assert restored.shape == (1, 200)
    torch.testing.assert_close(restored, source[:, 100:300], atol=4e-5, rtol=0)


def test_normalize_false_is_explicitly_rejected(tmp_path):
    path = tmp_path / "sample.wav"
    torchaudio.save(path, torch.zeros(1, 16), 16000)
    with pytest.raises(ValueError, match="normalize=True"):
        torchaudio.load(path, normalize=False)
