from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from nemotron_voicechat_runtime.eartts_acoustic_diagnostic import (
    AsyncEarTTSAcousticCapture,
)


def _wrapper(run_dir):
    return SimpleNamespace(
        _eartts_capture_dir=run_dir,
        _eartts_capture_sequence=0,
        _eartts_capture_metadata_sha256="test-metadata",
        _eartts_acoustic_diagnostic_raw_code=torch.tensor([[[101, 102]]]),
        _eartts_acoustic_diagnostic_policy={
            "current_subword_id": 5,
            "prior_content_tokens": 1,
            "prior_pad_tokens": 0,
            "tail_ratio": 3.0,
            "tail_done": False,
            "decoder_pad_policy_eligible": False,
            "decoder_pad_policy_applied": False,
        },
        _tts_in_turn_content=2,
        _tts_in_turn_pads=0,
        model=SimpleNamespace(
            tts_model=SimpleNamespace(text_bos_id=1, text_eos_id=2, text_pad_id=0)
        ),
    )


def _capture_position(capture, *, frame_index, code):
    capture.begin_call()
    capture.capture_frame(
        stream_id="stream",
        request_id="request",
        frame_index=frame_index,
        current_subword_id=torch.tensor([5]),
        previous_subword_id=torch.tensor([4]),
        current_subword_mask=torch.tensor([True]),
        raw_generated_code=code,
        recurrent_code=code + 10,
        decoder_code=code + 20,
        agent_idle_before=False,
        agent_idle_after=False,
    )


def test_async_capture_aligns_delayed_codec_and_final_drain(tmp_path):
    wrapper = _wrapper(tmp_path)
    capture = AsyncEarTTSAcousticCapture(wrapper)

    first_code = torch.tensor([[[1, 2]]])
    _capture_position(capture, frame_index=10, code=first_code)
    capture.finish_call(torch.empty(1, 0))

    wrapper._eartts_acoustic_diagnostic_raw_code = torch.tensor([[[201, 202]]])
    second_code = torch.tensor([[[3, 4]]])
    _capture_position(capture, frame_index=11, code=second_code)
    capture.finish_call(torch.full((1, 4), 0.25))
    capture.finalize_codec(torch.full((1, 4), 0.5), samples_per_frame=4)
    capture.close()

    first = torch.load(tmp_path / "frame-000000000.pt", weights_only=False)
    second = torch.load(tmp_path / "frame-000000001.pt", weights_only=False)
    assert first["frame_index"] == 10
    assert torch.equal(first["raw_generated_code"], torch.tensor([[[101, 102]]]))
    assert torch.equal(first["codec_pcm"], torch.full((1, 4), 0.25))
    assert first["codec_pcm_alignment"] == "next_wrapper_call_previous_submission"
    assert second["frame_index"] == 11
    assert torch.equal(second["raw_generated_code"], torch.tensor([[[201, 202]]]))
    assert torch.equal(second["codec_pcm"], torch.full((1, 4), 0.5))
    assert second["codec_pcm_alignment"] == "finalize_codec_drained_submission"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary == {
        "enqueued": 2,
        "placeholder_pcm_calls": 1,
        "queue_empty": True,
        "schema": "nemotron-live-eartts-acoustic-capture-summary-v1",
        "writer_alive": False,
        "writer_error": None,
        "written": 2,
    }
