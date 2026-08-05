import pytest

from nemotron_voicechat_runtime.cpu_codec_worker import load_codec_state


class FakeSafeOpen:
    def __init__(self, tensors):
        self.tensors = tensors

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, name):
        return self.tensors[name]


def test_load_codec_state_slices_public_combined_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    tensors = {
        "stt_model.ignore": 0,
        "tts_model.audio_codec.decoder.weight": 1,
        "tts_model.audio_codec.prvq.mus_list.0": 2,
    }
    state, details = load_codec_state(
        checkpoint,
        safe_open=FakeSafeOpen(tensors),
    )
    assert state == {"decoder.weight": 1, "prvq.mus_list.0": 2}
    assert details["layout"] == "public_combined"
    assert details["tensor_count"] == 2


def test_load_codec_state_refuses_missing_combined_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing combined model.safetensors"):
        load_codec_state(tmp_path, safe_open=FakeSafeOpen({}))
