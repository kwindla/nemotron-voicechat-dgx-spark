import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("torch", reason="runtime image supplies PyTorch")


RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "src" / "nemotron_voicechat_runtime"

from nemotron_voicechat_runtime.runtime_optimizations import (
    _finalize_pad_pair_call,
    _install_public_pad_pair_engine,
    _pad_pair_state,
    _record_pad_pair_decision,
    consume_pad_pair_trace,
    correct_add_fusion_pending,
    eartts_pad_should_decode_silence,
    enabled,
    finalize_public_fixed_stages,
    generate_next_token_delta,
    install_eartts_reset_on_bos,
    pad_pair_fc_in_progress,
    pad_pair_precontrol_reason,
    pad_pair_should_draft,
    slice_position_outputs,
)


def test_delta_generation_consumes_one_token_without_cumulative_history():
    import torch

    streaming_engine = pytest.importorskip(
        "nemo.collections.speechlm2.inference.vllm.streaming_llm_engine",
        reason="NVIDIA Speech is supplied by the runtime container",
    )
    from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import (  # noqa: E402
        RequestState,
        StreamStatus,
    )

    assert streaming_engine is not None

    class Outputs:
        def __init__(self):
            self.values = iter((7, 8))

        def __aiter__(self):
            return self

        async def __anext__(self):
            token = next(self.values)
            completion = SimpleNamespace(
                token_ids=[token], custom_outputs={"acoustic_tokens": token}
            )
            return SimpleNamespace(outputs=[completion], finished=False)

    class Backend:
        def __init__(self):
            self.outputs = Outputs()
            self.append_calls = 0

        def generate(self, *_args, **_kwargs):
            return self.outputs

        async def append_request(self, **_kwargs):
            self.append_calls += 1

    backend = Backend()
    state = RequestState("r", StreamStatus.ACTIVE, [], None)
    engine = SimpleNamespace(
        requests={"r": state},
        custom_input_specs=[SimpleNamespace(name="x", dtype="int32", dim=2)],
        engine=backend,
        sampling_params=object(),
        _get_safe_prompt_tokens=lambda length: [1] * length,
    )

    async def run():
        first = await generate_next_token_delta(engine, [torch.tensor([[1, 2]])], request_id="r")
        second = await generate_next_token_delta(engine, [torch.tensor([[3, 4]])], request_id="r")
        return first, second

    first, second = asyncio.run(run())

    assert (first.token_id, second.token_id) == (7, 8)
    assert state.generated_tokens == [7, 8]
    assert backend.append_calls == 1


def test_enabled_accepts_explicit_truthy_values_only():
    with mock.patch.dict("os.environ", {"FLAG": "yes"}):
        assert enabled("FLAG")
    with mock.patch.dict("os.environ", {"FLAG": "0"}):
        assert not enabled("FLAG")


def test_eartts_decoder_pad_silence_uses_prior_tail_counts():
    common = {
        "current_token": 12,
        "pad_token": 12,
        "agent_idle": False,
        "prior_content_tokens": 0,
        "tail_ratio": 3.0,
    }

    assert not eartts_pad_should_decode_silence(**common, prior_pad_tokens=0)
    assert eartts_pad_should_decode_silence(**common, prior_pad_tokens=1)
    assert eartts_pad_should_decode_silence(**{**common, "agent_idle": True}, prior_pad_tokens=0)
    assert not eartts_pad_should_decode_silence(
        **{**common, "current_token": 99}, prior_pad_tokens=10
    )


def test_eartts_reset_on_bos_restarts_only_eartts_and_uses_initial_code():
    import torch

    backend = mock.Mock()
    tts = mock.Mock()
    tts.text_bos_id = 1
    tts.tts_model = backend
    tts._reset_on_bos_installed = False
    original = mock.Mock(return_value=(torch.tensor([[[9]]]), None))
    tts.infer_codes_one_step = original
    wrapper = mock.Mock()
    wrapper.model.tts_model = tts
    wrapper.tts_init_inputs = {"speaker": torch.tensor([1])}
    wrapper.tts_prompt_token_ids = [7, 8]
    wrapper.first_tts_code_input = torch.tensor([[[3]]])

    assert install_eartts_reset_on_bos(wrapper)
    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=torch.tensor([[[99]]]),
        request_id="session-1",
    )

    backend.abort_request.assert_called_once_with("session-1")
    backend.assert_called_once_with(
        wrapper.tts_init_inputs,
        request_id="session-1",
        prompt_token_ids=[7, 8],
    )
    # The wrapped call must see the checkpoint's initial acoustic code, not the
    # previous turn's final code.
    assert original.call_args.kwargs["prev_audio_tokens"].tolist() == [[[3]]]


def test_pad_pair_fc_bypass_covers_active_and_forced_tool_states():
    assert not pad_pair_fc_in_progress(None)
    assert not pad_pair_fc_in_progress({"active": False})
    assert pad_pair_fc_in_progress({"active": True})
    assert pad_pair_fc_in_progress({"forced_function_tokens": [17]})
    assert pad_pair_fc_in_progress({"injecting_response": True})
    assert pad_pair_fc_in_progress({"awaiting_response": True})


def test_conditional_pad_pair_drafts_only_after_effective_pad():
    with mock.patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1"}, clear=True):
        assert not pad_pair_should_draft(None)
        assert not pad_pair_should_draft(False)
        assert pad_pair_should_draft(True)
    with mock.patch.dict("os.environ", {}, clear=True):
        assert pad_pair_should_draft(False)


def test_pad_pair_trace_is_opt_in_and_consumed():
    engine = SimpleNamespace(
        _voicechat_pad_pair_call_context={
            "request_id": "request-1",
            "frame_idx": 41,
            "engine_calls": 0,
            "bypass": False,
            "precontrol_reason": None,
        }
    )
    state = _pad_pair_state(engine, "request-1")
    state["previous_effective_pad"] = True
    wrapper = SimpleNamespace(model_llm_interface=SimpleNamespace(engine=engine))

    with mock.patch.dict("os.environ", {}, clear=True):
        _record_pad_pair_decision(
            engine,
            "request-1",
            state,
            decision="buffered",
            pending_before=False,
            bypass=False,
            should_draft=True,
            generated_before=50,
            generated_after=50,
        )
    assert not state["trace_events"]

    with mock.patch.dict(
        "os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True
    ):
        _record_pad_pair_decision(
            engine,
            "request-1",
            state,
            decision="buffered",
            pending_before=False,
            bypass=False,
            should_draft=True,
            generated_before=50,
            generated_after=50,
        )
        snapshot = consume_pad_pair_trace(wrapper, "request-1")

    assert [event["decision"] for event in snapshot["events"]] == ["buffered"]
    assert snapshot["events"][0]["pair_eligible"] is True
    assert snapshot["events"][0]["invariant_violation"] is False
    assert snapshot["events"][0]["request_id_match"] is True
    assert not state["trace_events"]


def test_pad_pair_trace_flags_request_mismatch_and_eligible_nonbuffer():
    engine = SimpleNamespace(
        _voicechat_pad_pair_call_context={
            "request_id": "wrapper-request",
            "frame_idx": 7,
            "engine_calls": 1,
            "bypass": False,
            "precontrol_reason": None,
        }
    )
    state = _pad_pair_state(engine, "engine-request")
    state["previous_effective_pad"] = True

    with mock.patch.dict(
        "os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True
    ):
        _record_pad_pair_decision(
            engine,
            "engine-request",
            state,
            decision="sequential_conditional",
            pending_before=False,
            bypass=False,
            should_draft=True,
            generated_before=8,
            generated_after=9,
        )

    event = state["trace_events"].popleft()
    assert event["call_origin"] == "additional_call_since_prepare"
    assert event["request_id_match"] is False
    assert event["invariant_violation"] is True


def test_pad_pair_trace_failure_cannot_discard_generation_result():
    engine = SimpleNamespace(
        _voicechat_pad_pair_call_context={
            "request_id": "request-1",
            "engine_calls": "not-an-integer",
        }
    )
    state = _pad_pair_state(engine, "request-1")

    with mock.patch.dict(
        "os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True
    ):
        _record_pad_pair_decision(
            engine,
            "request-1",
            state,
            decision="buffered",
            pending_before=False,
            bypass=False,
            should_draft=True,
            generated_before=1,
            generated_after=1,
        )

    assert state["trace_record_errors"] == 1
    assert not state["trace_events"]


def test_installed_pad_pair_scheduler_records_buffer_then_pair(monkeypatch):
    import torch

    streaming_engine = pytest.importorskip(
        "nemo.collections.speechlm2.inference.vllm.streaming_llm_engine",
        reason="NVIDIA Speech is supplied by the runtime container",
    )

    async def original(*_args, **_kwargs):
        raise AssertionError("buffered-to-pair path must not use the sequential backend")

    class LocalStreamingEngine:
        generate_next_token = original

    monkeypatch.setattr(streaming_engine, "LLMStreamingEngine", LocalStreamingEngine)
    _install_public_pad_pair_engine()

    class Iterator:
        async def __anext__(self):
            completion = SimpleNamespace(
                token_ids=[7, 12, 12],
                custom_outputs={"function_tokens": torch.tensor([12, 12])},
                finish_reason=None,
            )
            return SimpleNamespace(outputs=[completion], finished=False)

    class Backend:
        async def append_request(self, **_kwargs):
            return None

    request_state = SimpleNamespace(
        generation_iterator=Iterator(),
        generated_tokens=[7],
        status=object(),
    )
    engine = SimpleNamespace(
        _voicechat_pad_pair_token_id=12,
        _voicechat_pad_pair_call_context={
            "request_id": "request-1",
            "frame_idx": 10,
            "engine_calls": 0,
            "bypass": False,
            "precontrol_reason": None,
        },
        requests={"request-1": request_state},
        custom_input_specs=[{"name": "combined_embeds", "dtype": "float32"}],
        engine=Backend(),
    )
    state = _pad_pair_state(engine, "request-1")
    state["previous_effective_pad"] = True
    wrapper = SimpleNamespace(model_llm_interface=SimpleNamespace(engine=engine))

    async def run():
        first = await LocalStreamingEngine.generate_next_token(
            engine, [torch.zeros(1, 4)], request_id="request-1"
        )
        second = await LocalStreamingEngine.generate_next_token(
            engine, [torch.ones(1, 4)], request_id="request-1"
        )
        return first, second

    with mock.patch.dict(
        "os.environ",
        {
            "VOICECHAT_NANO_PAD_PAIR": "1",
            "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1",
            "VOICECHAT_NANO_PAD_PAIR_TRACE": "1",
        },
        clear=True,
    ):
        first, second = asyncio.run(run())
        snapshot = consume_pad_pair_trace(wrapper, "request-1")

    assert first.token_id == 12
    assert second.token_id == 12
    assert request_state.generated_tokens == [7, 12, 12]
    assert [event["decision"] for event in snapshot["events"]] == [
        "buffered",
        "pair_accepted",
    ]


def test_function_async_transition_has_latched_pending_drain_contract():
    source = (RUNTIME_ROOT / "runtime_optimizations.py").read_text()
    assert 'state["sequential_mode"] = True' in source
    assert 'state["sequential_pending_drains"] += 1' in source
    assert 'state["pending"] = None' in source


def test_control_transition_generalizes_pending_drain_latch():
    source = (RUNTIME_ROOT / "runtime_optimizations.py").read_text()
    assert 'enabled("VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER")' in source
    assert "control_tokens = {int(stt.text_bos_id), int(stt.text_eos_id)}" in source
    assert 'state["control_pending_drains"] += 1' in source


def test_control_transition_latches_a_live_pending_row():
    import torch

    pending = object()
    state = {
        "pending": pending,
        "needs_correction": False,
        "assumed": None,
        "previous_effective_pad": True,
        "sequential_mode": False,
        "control_barriers": 0,
        "control_pending_drains": 0,
    }
    engine = SimpleNamespace(
        _voicechat_pad_pair_token_id=12,
        _voicechat_pad_pair_states={"request-1": state},
        _voicechat_pad_pair_call_context={"request_id": "request-1"},
    )
    wrapper = SimpleNamespace(
        model=SimpleNamespace(stt_model=SimpleNamespace(text_bos_id=1, text_eos_id=2))
    )
    arguments = {
        "frame_idx": 3,
        "gen_text": torch.tensor([[12, 12, 12, 1]]),
        "gen_function_text": torch.tensor([[12, 12, 12, 12]]),
        "fc_state": None,
    }

    with mock.patch.dict(
        "os.environ",
        {
            "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "1",
            "VOICECHAT_NANO_PAD_PAIR_TRACE": "1",
        },
        clear=True,
    ):
        _finalize_pad_pair_call(wrapper, engine, arguments)

    assert state["pending"] is pending
    assert state["sequential_mode"] is True
    assert state["control_barriers"] == 1
    assert state["control_pending_drains"] == 1
    assert state["previous_effective_pad"] is False
    assert state["last_effective_tokens"] == {
        "frame_idx": 3,
        "request_id": "request-1",
        "text": 1,
        "function": 12,
        "pad": 12,
        "text_is_pad": False,
        "function_is_pad": True,
        "both_pad": False,
        "fc_in_progress": False,
        "precontrol_reason": None,
    }


def test_pad_pair_precontrol_predicts_rnnt_eou_one_frame_early():
    import torch

    wrapper = SimpleNamespace(
        _external_agent_eos_requested=False,
        _redirect_tokens_queue=[],
        _turn_taking_source="rnnt",
        _transport_input_activity_available=True,
        _transport_user_speech_seen=True,
        _transport_input_silent_frames=9,
        model_cfg={"force_turn_taking": True, "rnnt_eou_frames": 10},
    )
    arguments = {
        "rnnt_partial_hypotheses": {
            "agent_speaking": torch.tensor([False]),
            "blank_count": torch.tensor([9]),
            "speech_confirmed": torch.tensor([True]),
            "user_first_turn": torch.tensor([False]),
        }
    }

    assert pad_pair_precontrol_reason(wrapper, arguments) == "rnnt_eou"
    arguments["rnnt_partial_hypotheses"]["blank_count"][0] = 8
    assert pad_pair_precontrol_reason(wrapper, arguments) is None


def test_pad_pair_precontrol_covers_barge_in_external_eos_and_redirect():
    import torch

    wrapper = SimpleNamespace(
        _external_agent_eos_requested=False,
        _redirect_tokens_queue=[],
        _turn_taking_source="rnnt",
        model_cfg={
            "force_turn_taking": True,
            "rnnt_eou_frames": 10,
            "rnnt_bou_frames": 3,
        },
    )
    arguments = {
        "rnnt_partial_hypotheses": {
            "agent_speaking": torch.tensor([True]),
            "nonblank_consec": torch.tensor([2]),
        }
    }

    assert pad_pair_precontrol_reason(wrapper, arguments) == "rnnt_bou"
    wrapper._external_agent_eos_requested = True
    assert pad_pair_precontrol_reason(wrapper, arguments) == "external_eos"
    wrapper._external_agent_eos_requested = False
    wrapper._redirect_tokens_queue = [1]
    assert pad_pair_precontrol_reason(wrapper, arguments) == "redirect"


def test_pad_pair_selects_one_custom_output_row():
    import torch

    selected = slice_position_outputs(
        {
            "function_tokens": torch.tensor([12, 99]),
            "function_logits": torch.arange(6).reshape(2, 3),
        },
        1,
    )
    assert selected["function_tokens"].tolist() == [99]
    assert selected["function_logits"].tolist() == [[3, 4, 5]]


def test_pad_pair_corrects_text_and_function_add_fusion_channels():
    import torch

    embeddings = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    corrected = correct_add_fusion_pending(
        torch.zeros(1, 2),
        embed_tokens=lambda token_ids: embeddings[token_ids],
        assumed_text=0,
        effective_text=1,
        assumed_function=0,
        effective_function=2,
        agent_text_weight=1.0,
        function_weight=2.0,
    )
    assert corrected.tolist() == [[7.0, 10.0]]


def test_finalize_restores_stream_and_drains_codec():
    class CodecPipeline:
        submitted = 7
        worker_details = {"threads": 2}

        def finalize_audio(self, audio, *, output_device):
            assert output_device == "cuda"
            return "drained"

        def close(self):
            self.closed = True

    class Guard:
        def finish(self, stream_id):
            assert stream_id == 4
            return True

    bridged = object()
    codec = mock.Mock()
    codec._ea_cpu_bridged_decode = bridged
    wrapper = mock.Mock()
    wrapper.device = "cuda"
    wrapper._ea_async_codec_pipeline = CodecPipeline()
    wrapper._ea_realtime_gc_guard = Guard()
    wrapper.model.tts_model.audio_codec = codec
    state = mock.Mock(audio_buffer="buffered")

    result = finalize_public_fixed_stages(wrapper, 4, state)

    assert state.audio_buffer == "drained"
    assert codec.decode is bridged
    assert result["codec_submitted"] == 7
    assert result["cyclic_gc_restored"] is True
    assert not hasattr(wrapper, "_ea_async_codec_pipeline")


def test_finalize_retains_persistent_codec_and_resets_session():
    class CodecPipeline:
        submitted = 11
        worker_details = {"pid": 17}

        def finalize_audio(self, audio, *, output_device):
            assert output_device == "cuda"
            return "drained"

        def finish_session(self):
            return 3

        def close(self):
            raise AssertionError("persistent worker must not close per session")

    codec_pipeline = CodecPipeline()
    codec = mock.Mock()
    wrapper = mock.Mock()
    wrapper.device = "cuda"
    wrapper._ea_async_codec_pipeline = codec_pipeline
    wrapper.model.tts_model.audio_codec = codec
    state = mock.Mock(audio_buffer="buffered")

    with mock.patch.dict("os.environ", {"EA_CPU_CODEC_PERSISTENT": "1"}, clear=False):
        result = finalize_public_fixed_stages(wrapper, 5, state)

    assert state.audio_buffer == "drained"
    assert wrapper._ea_async_codec_pipeline is codec_pipeline
    assert result["codec_submitted"] == 3
    assert result["codec_persistent"] is True
