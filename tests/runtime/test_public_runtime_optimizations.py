import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("torch", reason="runtime image supplies PyTorch")


RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "src" / "nemotron_voicechat_runtime"

from nemotron_voicechat_runtime.runtime_optimizations import (
    _finalize_pad_pair_call,
    _install_public_pad_pair_engine,
    _pad_pair_buffered_custom_outputs,
    _pad_pair_state,
    _prepare_pad_pair_call,
    _record_pad_pair_decision,
    consume_pad_pair_trace,
    correct_add_fusion_pending,
    eartts_idle_pad_bypass_eligible,
    eartts_idle_pad_bypass_transition,
    eartts_pad_should_decode_silence,
    enabled,
    finalize_public_fixed_stages,
    generate_next_token_delta,
    install_eartts_decode_pad_silence,
    install_eartts_idle_pad_bypass,
    install_eartts_reset_on_bos,
    pad_pair_fc_in_progress,
    pad_pair_precontrol_reason,
    pad_pair_should_draft,
    slice_position_outputs,
)


def test_pad_pair_buffered_stamp_is_diagnostic_only() -> None:
    import torch

    current = torch.zeros(1, 4)
    with mock.patch.dict("os.environ", {}, clear=True):
        ordinary = _pad_pair_buffered_custom_outputs(current, 12)
    with mock.patch.dict(
        "os.environ",
        {"S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK": "20"},
        clear=True,
    ):
        diagnostic = _pad_pair_buffered_custom_outputs(current, 12)

    assert set(ordinary) == {"function_tokens"}
    assert set(diagnostic) == {
        "function_tokens",
        "voicechat_pad_pair_buffered",
    }
    assert diagnostic["voicechat_pad_pair_buffered"] is True
    assert ordinary["function_tokens"].tolist() == [12]

    with mock.patch.dict(
        "os.environ",
        {"S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK": "00"},
        clear=True,
    ):
        alternate_zero = _pad_pair_buffered_custom_outputs(current, 12)
    assert set(alternate_zero) == {"function_tokens"}


def test_retired_pad_pair_delegates_every_position_without_draft_state() -> None:
    import torch

    calls = []

    async def original(_engine, input_tensors, **_kwargs):
        calls.append(input_tensors[0].clone())
        return SimpleNamespace(token_id=12)

    class LocalStreamingEngine:
        generate_next_token = original

    streaming_engine = SimpleNamespace(
        LLMStreamingEngine=LocalStreamingEngine,
        GenerationResult=SimpleNamespace,
    )
    _install_public_pad_pair_engine(
        streaming_engine_module=streaming_engine,
        nemo_logging=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )
    engine = SimpleNamespace(
        _voicechat_pad_pair_token_id=12,
        custom_input_specs=[{"name": "combined_embeds", "dtype": "float32"}],
    )

    with mock.patch.dict(
        "os.environ",
        {
            "VOICECHAT_NANO_PAD_PAIR": "0",
            "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "0",
            "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "0",
        },
        clear=True,
    ):
        for index in range(4):
            result = asyncio.run(
                LocalStreamingEngine.generate_next_token(
                    engine,
                    [torch.full((1, 4), float(index))],
                    request_id="request-1",
                )
            )
            assert result.token_id == 12

    assert [int(value[0, 0].item()) for value in calls] == [0, 1, 2, 3]
    assert not hasattr(engine, "_voicechat_pad_pair_states")


def test_retained_stall_sequence_ends_at_the_next_packed_request() -> None:
    """Pin frames 20--23 of retained failing session 2d08e743 as executable evidence."""
    import torch

    module_name = "nemo.collections.speechlm2.inference.vllm.streaming_llm_engine"
    streaming_engine = ModuleType(module_name)
    streaming_engine.GenerationResult = SimpleNamespace
    streaming_engine.StreamStatus = SimpleNamespace(FINISHED=object())

    async def original(*_args, **_kwargs):
        raise AssertionError("confirmed PAD stretch must use the packed path")

    class LocalStreamingEngine:
        generate_next_token = original

    streaming_engine.LLMStreamingEngine = LocalStreamingEngine
    _install_public_pad_pair_engine(
        streaming_engine_module=streaming_engine,
        nemo_logging=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )

    class Iterator:
        def __init__(self, initial_tokens):
            self.initial_tokens = list(initial_tokens)
            self.calls = 0

        async def __anext__(self):
            self.calls += 1
            assert self.calls == 1
            accepted = self.initial_tokens + [12, 12]
            completion = SimpleNamespace(
                token_ids=accepted,
                custom_outputs={"function_tokens": torch.tensor([12, 12])},
                finish_reason=None,
            )
            return SimpleNamespace(outputs=[completion], finished=False)

    class PackedBoundaryReached(RuntimeError):
        pass

    class Backend:
        def __init__(self):
            self.append_shapes = []

        async def append_request(self, *, custom_inputs, **_kwargs):
            shape = tuple(custom_inputs["combined_embeds"].shape)
            self.append_shapes.append(shape)
            if len(self.append_shapes) == 2:
                raise PackedBoundaryReached("retained frame 23 enters packed Nano")

    generated = list(range(21))
    request_state = SimpleNamespace(
        generation_iterator=Iterator(generated),
        generated_tokens=generated,
        status=object(),
    )
    backend = Backend()
    engine = SimpleNamespace(
        _voicechat_pad_pair_token_id=12,
        requests={"request-1": request_state},
        custom_input_specs=[{"name": "combined_embeds", "dtype": "float32"}],
        engine=backend,
    )
    state = _pad_pair_state(engine, "request-1")
    state["previous_effective_pad"] = True

    async def run_trace_suffix():
        frame20 = await LocalStreamingEngine.generate_next_token(
            engine, [torch.full((1, 4), 20.0)], request_id="request-1"
        )
        frame20_position = len(request_state.generated_tokens)
        frame21 = await LocalStreamingEngine.generate_next_token(
            engine, [torch.full((1, 4), 21.0)], request_id="request-1"
        )
        frame21_position = len(request_state.generated_tokens)
        frame22 = await LocalStreamingEngine.generate_next_token(
            engine, [torch.full((1, 4), 22.0)], request_id="request-1"
        )
        frame22_position = len(request_state.generated_tokens)
        assert state["pending"] is not None
        with pytest.raises(PackedBoundaryReached, match="frame 23"):
            await LocalStreamingEngine.generate_next_token(
                engine, [torch.full((1, 4), 23.0)], request_id="request-1"
            )
        return frame20, frame21, frame22, (
            frame20_position,
            frame21_position,
            frame22_position,
        )

    with (
        mock.patch.dict(sys.modules, {module_name: streaming_engine}),
        mock.patch.dict(
            "os.environ",
            {
                "VOICECHAT_NANO_PAD_PAIR": "1",
                "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1",
            },
            clear=True,
        ),
    ):
        frame20, frame21, frame22, positions = asyncio.run(run_trace_suffix())

    assert positions == (21, 23, 23)
    assert (frame20.token_id, frame21.token_id, frame22.token_id) == (12, 12, 12)
    assert backend.append_shapes == [(2, 4), (2, 4)]


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
            self.generate_request_ids = []
            self.append_request_ids = []

        def generate(self, *_args, **kwargs):
            self.generate_request_ids.append(kwargs["request_id"])
            return self.outputs

        async def append_request(self, **kwargs):
            self.append_calls += 1
            self.append_request_ids.append(kwargs["request_id"])

    backend = Backend()
    state = RequestState(
        "r",
        StreamStatus.ACTIVE,
        [],
        None,
        backend_request_id="r__eartts_generation_7",
        backend_generation=7,
    )
    engine = SimpleNamespace(
        requests={"r": state},
        custom_input_specs=[SimpleNamespace(name="x", dtype="int32", dim=2)],
        engine=backend,
        sampling_params=object(),
        _get_safe_prompt_tokens=lambda length: [1] * length,
        resolve_backend_request_id=lambda logical: (
            "r__eartts_generation_7" if logical == "r" else logical
        ),
    )

    async def run():
        first = await generate_next_token_delta(engine, [torch.tensor([[1, 2]])], request_id="r")
        second = await generate_next_token_delta(engine, [torch.tensor([[3, 4]])], request_id="r")
        return first, second

    first, second = asyncio.run(run())

    assert (first.token_id, second.token_id) == (7, 8)
    assert state.generated_tokens == [7, 8]
    assert backend.append_calls == 1
    assert backend.generate_request_ids == ["r__eartts_generation_7"]
    assert backend.append_request_ids == ["r__eartts_generation_7"]


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


def test_eartts_idle_pad_bypass_is_narrow_and_fc_fail_closed():
    common = {
        "enabled": True,
        "decode_audio": True,
        "use_vllm_eartts": True,
        "ordinary_pcm": True,
        "source_embeddings_supplied": False,
        "force_output_pad": False,
        "current_token": 12,
        "pad_token": 12,
        "agent_idle": True,
        "fc_state": None,
        "acoustic_capture_enabled": False,
    }
    assert eartts_idle_pad_bypass_eligible(**common)
    for mutation in (
        {"enabled": False},
        {"decode_audio": False},
        {"use_vllm_eartts": False},
        {"ordinary_pcm": False},
        {"source_embeddings_supplied": True},
        {"force_output_pad": True},
        {"current_token": 1},
        {"agent_idle": False},
        {"acoustic_capture_enabled": True},
        {"fc_state": {"active": True}},
        {"fc_state": {"awaiting_response": True}},
        {"fc_state": {"awaiting_eotr": True}},
        {"fc_state": {"injecting_response": True}},
        {"fc_state": {"forced_function_tokens": [7]}},
    ):
        assert not eartts_idle_pad_bypass_eligible(**{**common, **mutation})
    assert eartts_idle_pad_bypass_eligible(
        **{**common, "fc_state": {"active": False, "forced_function_tokens": []}}
    )


def test_eartts_idle_pad_bypass_transition_preserves_recurrent_identity():
    import torch

    recurrent = torch.tensor([[[9, 8, 7]]])
    pkv = object()
    silence = torch.tensor([1, 2, 3])

    decoder, preserved_recurrent, preserved_pkv = eartts_idle_pad_bypass_transition(
        recurrent_code=recurrent,
        past_key_values=pkv,
        codec_silence_tokens=silence,
    )

    assert decoder.tolist() == [[[1, 2, 3]]]
    assert decoder.data_ptr() != silence.data_ptr()
    assert preserved_recurrent is recurrent
    assert preserved_pkv is pkv


def test_eartts_idle_pad_bypass_matches_decoder_policy_and_bos_resynchronizes():
    import torch

    backend = mock.Mock()
    generated = torch.tensor([[[9, 8, 7]]])
    original = mock.Mock(return_value=(generated, "advanced-pkv"))
    tts = SimpleNamespace(
        infer_codes_one_step=original,
        text_pad_id=12,
        text_bos_id=1,
        codec_silence_tokens=torch.tensor([1, 2, 3]),
        tts_model=backend,
        _decode_pad_silence_installed=False,
        _reset_on_bos_installed=False,
    )
    wrapper = SimpleNamespace(
        model=SimpleNamespace(tts_model=tts),
        use_vllm_eartts=True,
        _get_agent_idle=lambda: True,
        _tts_in_turn_content=0,
        _tts_in_turn_pads=0,
        tts_init_inputs={"speaker": torch.tensor([1])},
        tts_prompt_token_ids=[7, 8],
        first_tts_code_input=torch.tensor([[[3, 3, 3]]]),
    )
    assert install_eartts_decode_pad_silence(wrapper)
    baseline_decoder, _ = tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[12]])
    )
    assert install_eartts_idle_pad_bypass(wrapper)
    recurrent = torch.tensor([[[6, 5, 4]]])
    pkv = object()
    bypass_decoder, recurrent_after, pkv_after = (
        wrapper._eartts_idle_pad_bypass_transition(
            recurrent_code=recurrent,
            past_key_values=pkv,
            codec_silence_tokens=tts.codec_silence_tokens,
        )
    )
    assert torch.equal(bypass_decoder, baseline_decoder)
    assert recurrent_after is recurrent
    assert pkv_after is pkv
    baseline_history = torch.tensor([[[4, 4, 4], [5, 5, 5]]])
    assert torch.equal(
        torch.cat([baseline_history[:, 1:], bypass_decoder], dim=1),
        torch.cat([baseline_history[:, 1:], baseline_decoder], dim=1),
    )

    assert install_eartts_reset_on_bos(wrapper)
    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=recurrent_after,
        past_key_values=pkv_after,
        request_id="session-1",
    )
    backend.abort_request.assert_called_once_with("session-1")
    assert original.call_args.kwargs["prev_audio_tokens"].tolist() == [[[3, 3, 3]]]


def test_eartts_idle_pad_bypass_install_requires_decoder_silence_policy():
    wrapper = SimpleNamespace(
        use_vllm_eartts=True,
        first_tts_code_input=None,
        model=SimpleNamespace(
            tts_model=SimpleNamespace(
                _decode_pad_silence_installed=False,
                codec_silence_tokens=object(),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="decoder-silence policy"):
        install_eartts_idle_pad_bypass(wrapper)

    wrapper.model.tts_model._decode_pad_silence_installed = True
    torch = pytest.importorskip("torch")
    wrapper.first_tts_code_input = torch.zeros(1, 1, 2)
    wrapper.model.tts_model.codec_silence_tokens = torch.zeros(1)
    with pytest.raises(RuntimeError, match="shape does not match"):
        install_eartts_idle_pad_bypass(wrapper)
    wrapper.first_tts_code_input = torch.zeros(1, 1, 1)
    assert install_eartts_idle_pad_bypass(wrapper)
    assert wrapper._eartts_idle_pad_bypass_installed is True
    assert wrapper._eartts_idle_pad_bypass_eligible is eartts_idle_pad_bypass_eligible
    assert wrapper._eartts_idle_pad_bypass_transition is eartts_idle_pad_bypass_transition


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
    with mock.patch(
        "nemotron_voicechat_runtime.runtime_optimizations.time.perf_counter",
        side_effect=[10.0, 10.002, 10.007],
    ):
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
    assert tts._reset_on_bos_count == 1
    assert tts._reset_on_bos_last_request_id == "session-1"
    assert tts._reset_on_bos_last_timing_ms == {
        "schema": 1,
        "reset_count": 1,
        "request_id": "session-1",
        "abort_call_ms": pytest.approx(2.0),
        "prefill_call_ms": pytest.approx(5.0),
        "reset_total_ms": pytest.approx(7.0),
    }

    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[12]]),
        prev_audio_tokens=torch.tensor([[[3]]]),
        request_id="session-1",
    )
    assert tts._reset_on_bos_last_timing_ms is None


def test_eartts_failed_bos_reset_does_not_leak_prior_timing():
    import torch

    backend = mock.Mock()
    tts = mock.Mock()
    tts.text_bos_id = 1
    tts.tts_model = backend
    tts._reset_on_bos_installed = False
    original = mock.Mock(side_effect=RuntimeError("infer failed"))
    tts.infer_codes_one_step = original
    wrapper = mock.Mock()
    wrapper.model.tts_model = tts
    wrapper.tts_init_inputs = {"speaker": torch.tensor([1])}
    wrapper.tts_prompt_token_ids = [7, 8]
    wrapper.first_tts_code_input = torch.tensor([[[3]]])
    assert install_eartts_reset_on_bos(wrapper)
    tts._reset_on_bos_last_timing_ms = {"stale": True}

    with (
        mock.patch(
            "nemotron_voicechat_runtime.runtime_optimizations.time.perf_counter",
            side_effect=[10.0, 10.002, 10.007],
        ),
        pytest.raises(RuntimeError, match="infer failed"),
    ):
        tts.infer_codes_one_step(
            current_subword_id=torch.tensor([[1]]),
            prev_audio_tokens=torch.tensor([[[99]]]),
            request_id="session-1",
        )

    assert tts._reset_on_bos_last_timing_ms is None


def _prepared_epoch_fixture():
    import torch

    request_id = "session-1"
    request = SimpleNamespace(
        status=SimpleNamespace(value="active"),
        generated_tokens=[4],
        generation_iterator=object(),
        backend_request_id=f"{request_id}__eartts_generation_1",
        backend_generation=1,
    )

    class Backend:
        def __init__(self):
            self.engine = SimpleNamespace(
                requests={request_id: request},
                _unique_backend_request_ids=True,
                resolve_backend_request_id=lambda value: self.engine.requests[
                    value
                ].backend_request_id,
            )
            self.abort_calls = []
            self.prefill_calls = []
            self.fail_prefill = False
            self.backend_generation = 1

        def abort_request(self, value):
            self.abort_calls.append(value)
            return self.engine.requests.pop(value, None) is not None

        def __call__(self, inputs, *, request_id, prompt_token_ids):
            self.prefill_calls.append((inputs, request_id, prompt_token_ids))
            if self.fail_prefill:
                self.fail_prefill = False
                raise RuntimeError("prefill failed")
            self.backend_generation += 1
            self.engine.requests[request_id] = SimpleNamespace(
                status=SimpleNamespace(value="active"),
                generated_tokens=[8],
                generation_iterator=object(),
                backend_request_id=(
                    f"{request_id}__eartts_generation_{self.backend_generation}"
                ),
                backend_generation=self.backend_generation,
            )
            return SimpleNamespace()

    backend = Backend()
    original = mock.Mock(return_value=(torch.tensor([[[9]]]), "pkv"))
    tts = SimpleNamespace(
        text_bos_id=1,
        tts_model=backend,
        infer_codes_one_step=original,
        _reset_on_bos_installed=False,
    )
    wrapper = SimpleNamespace(
        model=SimpleNamespace(tts_model=tts),
        tts_init_inputs={"speaker": torch.tensor([1])},
        tts_prompt_token_ids=[7, 8],
        first_tts_code_input=torch.tensor([[[3]]]),
    )
    assert install_eartts_reset_on_bos(wrapper, prepared_epoch=True)
    return wrapper, tts, backend, original, request_id


def test_prepared_epoch_is_unarmed_until_commit_and_consumed_before_bos():
    import torch

    wrapper, tts, backend, original, request_id = _prepared_epoch_fixture()
    attested = wrapper._attest_eartts_prepared_epoch(request_id)
    assert attested["state"] == "clean_unarmed"
    assert wrapper._arm_eartts_prepared_epoch(
        request_id, 11, quiescent_check=lambda: True
    )["passed"] is True
    assert wrapper._eartts_prepared_epoch_status()["state"] == "clean_armed"

    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=torch.tensor([[[99]]]),
        request_id=request_id,
    )

    assert backend.abort_calls == []
    assert backend.prefill_calls == []
    assert original.call_args.kwargs["prev_audio_tokens"].tolist() == [[[3]]]
    status = wrapper._eartts_prepared_epoch_status()
    assert status["state"] == "dirty"
    assert status["armed_client_turn_id"] is None
    assert status["prepared_reuse_count"] == 1
    assert status["bos_transition_count"] == 1
    assert status["last_bos_event"]["mode"] == "prepared_reuse"
    assert status["last_bos_event"]["armed_client_turn_id"] == 11
    assert tts._reset_on_bos_count == 0


def test_prepared_epoch_new_turn_clears_prior_bos_event_before_frames_resume():
    import torch

    wrapper, tts, _backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    assert wrapper._arm_eartts_prepared_epoch(
        request_id, 1, quiescent_check=lambda: True
    )["passed"] is True
    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=torch.tensor([[[99]]]),
        request_id=request_id,
    )
    prior = wrapper._eartts_prepared_epoch_status()
    assert prior["last_bos_event"]["prepare_epoch"] == prior["prepare_epoch"]

    prepared = wrapper._prepare_eartts_prepared_epoch(
        request_id, quiescent_check=lambda: True
    )

    assert prepared["passed"] is True
    assert prepared["state"] == "clean_unarmed"
    assert prepared["prepare_epoch"] == prior["prepare_epoch"] + 1
    assert prepared["last_bos_event"] is None


def test_prepared_epoch_transport_reset_zeros_all_session_counters():
    import torch

    wrapper, tts, _backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    assert wrapper._arm_eartts_prepared_epoch(
        request_id, 1, quiescent_check=lambda: True
    )["passed"] is True
    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=torch.tensor([[[99]]]),
        request_id=request_id,
    )
    controller = wrapper._eartts_prepared_epoch_controller
    controller.prepare_count = 2
    controller.prepare_failure_count = 1
    tts._reset_on_bos_count = 3
    tts._reset_on_bos_last_request_id = request_id
    tts._reset_on_bos_last_timing_ms = {"schema": 1}

    wrapper._reset_eartts_prepared_epoch("transport_session_reset")

    status = wrapper._eartts_prepared_epoch_status()
    assert status["state"] == "invalid"
    assert status["reason"] == "transport_session_reset"
    assert status["prepare_count"] == 0
    assert status["prepare_failure_count"] == 0
    assert status["prepared_reuse_count"] == 0
    assert status["bos_transition_count"] == 0
    assert status["last_bos_event"] is None
    assert tts._reset_on_bos_count == 0
    assert tts._reset_on_bos_last_request_id is None
    assert tts._reset_on_bos_last_timing_ms is None


def test_prepared_epoch_real_call_invalidates_and_bos_falls_back_to_reset():
    import torch

    wrapper, tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[12]]),
        prev_audio_tokens=torch.tensor([[[3]]]),
        request_id=request_id,
    )
    assert wrapper._eartts_prepared_epoch_status()["state"] == "dirty"
    assert wrapper._arm_eartts_prepared_epoch(
        request_id, 3, quiescent_check=lambda: True
    )["passed"] is False

    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=torch.tensor([[[99]]]),
        request_id=request_id,
    )
    status = wrapper._eartts_prepared_epoch_status()
    assert backend.abort_calls == [request_id]
    assert len(backend.prefill_calls) == 1
    assert tts._reset_on_bos_count == 1
    assert status["prepared_reuse_count"] == 0
    assert status["bos_transition_count"] == 1
    assert status["last_bos_event"]["mode"] == "legacy_fallback"


def test_prepared_epoch_clean_but_unarmed_bos_uses_legacy_reset():
    import torch

    wrapper, tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=torch.tensor([[[99]]]),
        request_id=request_id,
    )
    status = wrapper._eartts_prepared_epoch_status()
    assert backend.abort_calls == [request_id]
    assert tts._reset_on_bos_count == 1
    assert status["prepared_reuse_count"] == 0
    assert status["last_bos_event"]["mode"] == "legacy_fallback"
    assert status["last_bos_event"]["fallback_reason"] == "prepared_epoch_not_armed"


def test_prepared_epoch_arm_is_consumed_before_failed_bos_inference():
    import torch

    wrapper, tts, _backend, original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    assert wrapper._arm_eartts_prepared_epoch(
        request_id, 4, quiescent_check=lambda: True
    )["passed"] is True
    original.side_effect = RuntimeError("BOS failed")
    with pytest.raises(RuntimeError, match="BOS failed"):
        tts.infer_codes_one_step(
            current_subword_id=torch.tensor([[1]]),
            prev_audio_tokens=torch.tensor([[[99]]]),
            request_id=request_id,
        )
    status = wrapper._eartts_prepared_epoch_status()
    assert status["state"] == "invalid"
    assert status["armed_client_turn_id"] is None
    assert status["prepared_reuse_count"] == 0
    assert status["bos_transition_count"] == 0
    assert status["last_bos_event"] is None


def test_prepared_epoch_prepare_failure_stays_invalid_until_legacy_bos_recovers():
    import torch

    wrapper, tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    controller = wrapper._eartts_prepared_epoch_controller
    controller.mark_real_call("test_dirty")
    backend.fail_prefill = True

    failed = wrapper._prepare_eartts_prepared_epoch(
        request_id, quiescent_check=lambda: True
    )
    assert failed["passed"] is False
    assert failed["state"] == "invalid"
    repeated = wrapper._prepare_eartts_prepared_epoch(
        request_id, quiescent_check=lambda: True
    )
    assert repeated["passed"] is False
    assert repeated["state"] == "invalid"
    tts.infer_codes_one_step(
        current_subword_id=torch.tensor([[1]]),
        prev_audio_tokens=torch.tensor([[[99]]]),
        request_id=request_id,
    )
    status = wrapper._eartts_prepared_epoch_status()
    assert status["state"] == "dirty"
    assert status["last_bos_event"]["mode"] == "legacy_fallback"
    assert backend.abort_calls == [request_id, request_id]
    assert len(backend.prefill_calls) == 2


def test_prepared_epoch_request_mutation_rejects_arm_and_session_reset_invalidates():
    wrapper, _tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    backend.engine.requests[request_id].generated_tokens.append(9)
    rejected = wrapper._arm_eartts_prepared_epoch(
        request_id, 1, quiescent_check=lambda: True
    )
    assert rejected["passed"] is False
    assert rejected["reason"] == "generated_tokens_mismatch"
    session_before = rejected["session_epoch"]
    wrapper._reset_eartts_prepared_epoch("test_reset")
    status = wrapper._eartts_prepared_epoch_status()
    assert status["session_epoch"] == session_before + 1
    assert status["state"] == "invalid"
    assert status["armed_client_turn_id"] is None


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("request_state", "request_state_id_mismatch"),
        ("iterator", "iterator_id_mismatch"),
        ("backend_request_id", "backend_request_id_mismatch"),
        ("backend_generation", "backend_generation_mismatch"),
        ("prompt", "prompt_token_ids_mismatch"),
        ("speaker", "speaker_identity_mismatch"),
        ("initial_code", "initial_code_identity_mismatch"),
    ),
)
def test_prepared_epoch_identity_mutations_fail_closed(mutation, reason):
    import torch

    wrapper, _tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    request = backend.engine.requests[request_id]
    if mutation == "request_state":
        backend.engine.requests[request_id] = SimpleNamespace(
            status=SimpleNamespace(value="active"),
            generated_tokens=list(request.generated_tokens),
            generation_iterator=request.generation_iterator,
            backend_request_id=request.backend_request_id,
            backend_generation=request.backend_generation,
        )
    elif mutation == "iterator":
        request.generation_iterator = object()
    elif mutation == "backend_request_id":
        request.backend_request_id = f"{request_id}__eartts_generation_999"
    elif mutation == "backend_generation":
        request.backend_generation += 1
    elif mutation == "prompt":
        wrapper.tts_prompt_token_ids.append(9)
    elif mutation == "speaker":
        wrapper.tts_init_inputs["speaker"].add_(torch.tensor([1]))
    elif mutation == "initial_code":
        wrapper.first_tts_code_input.add_(torch.tensor([[[1]]]))
    rejected = wrapper._arm_eartts_prepared_epoch(
        request_id, 1, quiescent_check=lambda: True
    )
    assert rejected["passed"] is False
    assert rejected["reason"] == reason


def test_prepared_epoch_false_abort_never_publishes_clean_snapshot():
    wrapper, _tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    controller = wrapper._eartts_prepared_epoch_controller
    controller.mark_real_call("test_dirty")
    backend.abort_request = mock.Mock(return_value=False)

    evidence = wrapper._prepare_eartts_prepared_epoch(
        request_id, quiescent_check=lambda: True
    )
    assert evidence["passed"] is False
    assert evidence["state"] == "invalid"
    assert backend.prefill_calls == []


def test_prepared_epoch_rechecks_quiescence_and_retains_completed_prefill_timing():
    wrapper, _tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    wrapper._eartts_prepared_epoch_controller.mark_real_call("test_dirty")
    quiescence = iter((True, False))

    evidence = wrapper._prepare_eartts_prepared_epoch(
        request_id, quiescent_check=lambda: next(quiescence)
    )
    assert evidence["passed"] is False
    assert evidence["state"] == "invalid"
    assert evidence["timing"]["prefill_call_ms"] is not None
    assert len(backend.prefill_calls) == 1


def test_prepared_epoch_already_clean_still_requires_function_quiescence():
    wrapper, _tts, backend, _original, request_id = _prepared_epoch_fixture()
    wrapper._attest_eartts_prepared_epoch(request_id)
    evidence = wrapper._prepare_eartts_prepared_epoch(
        request_id, quiescent_check=lambda: False
    )
    assert evidence["passed"] is False
    assert evidence["reason"] == "function_cycle_not_quiescent"
    assert backend.abort_calls == []


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

    with mock.patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True):
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

    with mock.patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True):
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

    with mock.patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True):
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


@pytest.mark.parametrize(
    ("decision", "iterator_ready", "pending", "previous_pad", "sequential"),
    [
        ("initial", False, False, None, False),
        ("sequential_bypass", True, False, None, True),
        ("sequential_conditional", True, False, False, False),
        ("sequential_pending_drain", True, True, True, True),
        ("sequential_pending_roll", True, True, False, False),
    ],
)
def test_pad_pair_nonbuffered_scheduler_paths_never_stamp_synthetic_marker(
    decision,
    iterator_ready,
    pending,
    previous_pad,
    sequential,
):
    import torch

    async def original(engine, _input_tensors, **_kwargs):
        engine.requests["request-1"].generated_tokens.append(31)
        return SimpleNamespace(
            token_id=31,
            custom_outputs={"function_tokens": torch.tensor([31])},
        )

    class LocalStreamingEngine:
        generate_next_token = original

    streaming_engine = SimpleNamespace(
        LLMStreamingEngine=LocalStreamingEngine,
        GenerationResult=SimpleNamespace,
    )
    _install_public_pad_pair_engine(
        streaming_engine_module=streaming_engine,
        nemo_logging=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )
    request_state = SimpleNamespace(
        generation_iterator=object() if iterator_ready else None,
        generated_tokens=[7] if iterator_ready else [],
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
    )
    state = _pad_pair_state(engine, "request-1")
    state["previous_effective_pad"] = previous_pad
    state["sequential_mode"] = sequential
    state["pending"] = torch.ones(1, 4) if pending else None
    wrapper = SimpleNamespace(model_llm_interface=SimpleNamespace(engine=engine))

    with mock.patch.dict(
        "os.environ",
        {
            "VOICECHAT_NANO_PAD_PAIR": "1",
            "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1",
            "VOICECHAT_NANO_PAD_PAIR_TRACE": "1",
            "S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK": "20",
        },
        clear=True,
    ):
        result = asyncio.run(
            LocalStreamingEngine.generate_next_token(
                engine,
                [torch.zeros(1, 4)],
                request_id="request-1",
            )
        )
        snapshot = consume_pad_pair_trace(wrapper, "request-1")

    assert "voicechat_pad_pair_buffered" not in result.custom_outputs
    assert [event["decision"] for event in snapshot["events"]] == [decision]


@pytest.mark.parametrize(
    ("packed_token_ids", "packed_function_tokens", "expected_decision"),
    [
        ([7, 12, 12], [12, 12], "pair_accepted"),
        ([7, 31], [31], "pair_rejected"),
    ],
)
def test_installed_pad_pair_scheduler_records_buffer_then_pair(
    monkeypatch,
    packed_token_ids,
    packed_function_tokens,
    expected_decision,
):
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
                token_ids=packed_token_ids,
                custom_outputs={
                    "function_tokens": torch.tensor(packed_function_tokens)
                },
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
            "S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK": "20",
        },
        clear=True,
    ):
        first, second = asyncio.run(run())
        snapshot = consume_pad_pair_trace(wrapper, "request-1")

    assert first.token_id == 12
    assert second.token_id == packed_token_ids[-1]
    assert first.custom_outputs["voicechat_pad_pair_buffered"] is True
    assert "voicechat_pad_pair_buffered" not in second.custom_outputs
    assert request_state.generated_tokens == packed_token_ids
    assert [event["decision"] for event in snapshot["events"]] == [
        "buffered",
        expected_decision,
    ]


def test_external_user_eou_precontrol_drains_pending_row_sequentially():
    import torch

    calls = []

    async def original(engine, input_tensors, **_kwargs):
        calls.append(input_tensors[0].clone())
        engine.requests["request-1"].generated_tokens.append(31)
        return SimpleNamespace(token_id=31)

    class LocalStreamingEngine:
        generate_next_token = original

    class AddFusion:
        pass

    streaming_engine = SimpleNamespace(
        LLMStreamingEngine=LocalStreamingEngine,
        GenerationResult=SimpleNamespace,
    )
    _install_public_pad_pair_engine(
        streaming_engine_module=streaming_engine,
        nemo_logging=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )

    engine = SimpleNamespace(
        _voicechat_pad_pair_token_id=12,
        requests={
            "request-1": SimpleNamespace(
                generation_iterator=object(),
                generated_tokens=[7],
            )
        },
        custom_input_specs=[{"name": "combined_embeds", "dtype": "float32"}],
    )
    wrapper = SimpleNamespace(
        _external_user_eou_requested=True,
        _external_agent_eos_requested=False,
        _redirect_tokens_queue=[],
        _turn_taking_source="rnnt",
        _has_asr_head=False,
        request_id="request-1",
        fusion_module=AddFusion(),
        model_llm_interface=SimpleNamespace(engine=engine),
        model=SimpleNamespace(
            stt_model=SimpleNamespace(text_bos_id=1, text_eos_id=2),
        ),
    )
    pending = torch.ones(1, 4)
    current = torch.full((1, 4), 2.0)
    state = _pad_pair_state(engine, "request-1")
    state["pending"] = pending
    state["previous_effective_pad"] = True
    arguments = {
        "frame_idx": 1,
        "num_frames_per_chunk": 1,
        "gen_text": torch.tensor([[12, 12]]),
        "gen_function_text": torch.tensor([[12, 12]]),
        "fc_state": None,
    }

    async def run():
        _prepare_pad_pair_call(wrapper, arguments)
        first = await LocalStreamingEngine.generate_next_token(
            engine,
            [current],
            request_id="request-1",
        )
        # NVIDIA's turn-taking hook consumes the latch and overwrites this
        # position with BOS after Nano returns, before the optimizer finalizes.
        wrapper._external_user_eou_requested = False
        arguments["gen_text"][0, 1] = 1
        _finalize_pad_pair_call(wrapper, engine, arguments)

        after_bos = torch.full((1, 4), 3.0)
        next_arguments = {
            **arguments,
            "frame_idx": 2,
            "gen_text": torch.tensor([[12, 1, 12]]),
            "gen_function_text": torch.tensor([[12, 12, 12]]),
        }
        _prepare_pad_pair_call(wrapper, next_arguments)
        second = await LocalStreamingEngine.generate_next_token(
            engine,
            [after_bos],
            request_id="request-1",
        )
        return first, second, after_bos

    with mock.patch.dict(
        "os.environ",
        {
            "VOICECHAT_NANO_PAD_PAIR": "1",
            "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1",
            "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "1",
        },
        clear=True,
    ):
        first, second, after_bos = asyncio.run(run())

    assert first.token_id == 31
    assert second.token_id == 31
    assert len(calls) == 2
    assert torch.equal(calls[0], pending)
    assert not torch.equal(calls[0], current)
    assert torch.equal(calls[1], after_bos)
    assert state["pending"] is None
    assert state["sequential_mode"] is False
    assert state["sequential_pending_drains"] == 1
    assert state["precontrol_pending_drains"] == 1
    assert state["previous_effective_pad"] is False


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
        _external_user_eou_requested=False,
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


def test_pad_pair_precontrol_covers_external_user_eou_barge_in_and_redirect():
    import torch

    wrapper = SimpleNamespace(
        _external_user_eou_requested=False,
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
    wrapper._external_user_eou_requested = True
    assert pad_pair_precontrol_reason(wrapper, arguments) == "external_user_eou"
    wrapper._external_user_eou_requested = False
    wrapper._external_agent_eos_requested = True
    assert pad_pair_precontrol_reason(wrapper, arguments) == "external_eos"
    wrapper._external_agent_eos_requested = False
    wrapper._redirect_tokens_queue = [1]
    assert pad_pair_precontrol_reason(wrapper, arguments) == "redirect"


def test_external_user_eou_precontrol_arms_pending_drain_before_nano_step():
    import torch

    class AddFusion:
        pass

    engine = SimpleNamespace(_voicechat_pad_pair_token_id=12)
    wrapper = SimpleNamespace(
        _external_user_eou_requested=True,
        _external_agent_eos_requested=False,
        _redirect_tokens_queue=[],
        _turn_taking_source="rnnt",
        _has_asr_head=False,
        request_id="request-1",
        fusion_module=AddFusion(),
        model_llm_interface=SimpleNamespace(engine=engine),
    )
    state = _pad_pair_state(engine, "request-1")
    pending = torch.ones(1, 4)
    state["pending"] = pending
    state["previous_effective_pad"] = True
    arguments = {
        "frame_idx": 1,
        "num_frames_per_chunk": 1,
        "gen_text": torch.tensor([[12, 12]]),
        "gen_function_text": torch.tensor([[12, 12]]),
        "fc_state": None,
    }

    with mock.patch.dict(
        "os.environ",
        {
            "VOICECHAT_NANO_PAD_PAIR": "1",
            "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "1",
        },
        clear=True,
    ):
        prepared = _prepare_pad_pair_call(wrapper, arguments)

    assert prepared is engine
    assert state["pending"] is pending
    assert state["sequential_mode"] is True
    assert state["precontrol_barriers"] == 1
    assert state["precontrol_pending_drains"] == 1
    assert state["precontrol_reasons"] == {"external_user_eou": 1}
    assert engine._voicechat_pad_pair_call_context["precontrol_reason"] == "external_user_eou"


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


def test_installed_function_logit_trace_is_bounded_and_argmax_consistent() -> None:
    import torch

    model_factory = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.model_factory",
        reason="NVIDIA Speech is supplied by the runtime container",
    )
    interface = model_factory.VllmLLMModel.__new__(model_factory.VllmLLMModel)
    interface._function_logit_trace_topk = 2
    interface._function_logit_trace_ids = {
        "pad": 0,
        "sotc": 1,
        "eotc": 2,
        "eotr": 3,
        "text_bos": 4,
    }
    trace = interface._summarize_function_logits(
        torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]]), torch.tensor([0])
    )

    assert trace["head_argmax_id"] == trace["predicted_token_id"] == 0
    assert trace["head_argmax_logit"] == 5.0
    assert trace["head_argmax_equal_count"] == 1
    assert trace["sotc_rank"] == 2
    assert [item["token_id"] for item in trace["top_k"]] == [0, 1]
    assert set(trace["watched"]) == {"pad", "sotc", "eotc", "eotr", "text_bos"}
    assert all(not hasattr(value, "detach") for value in trace.values())

    with pytest.raises(RuntimeError, match="argmax does not match"):
        interface._summarize_function_logits(
            torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]]), torch.tensor([1])
        )
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        interface._summarize_function_logits(
            torch.tensor([[float("nan"), 4.0, 3.0, 2.0, 1.0]]), torch.tensor([0])
        )
    with pytest.raises(RuntimeError, match="function_logits custom output is missing"):
        interface._attach_function_logit_trace(
            {"function_predicted_token": torch.tensor([0])},
            SimpleNamespace(custom_outputs={"function_tokens": torch.tensor([0])}),
        )


def test_installed_agent_logit_trace_is_bounded_and_argmax_consistent() -> None:
    import torch

    model_factory = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.model_factory",
        reason="NVIDIA Speech is supplied by the runtime container",
    )
    interface = model_factory.VllmLLMModel.__new__(model_factory.VllmLLMModel)
    interface._agent_logit_trace_topk = 3
    interface._agent_logit_trace_ids = {"pad": 0, "bos": 1}
    interface._text_pad_id = 0
    trace = interface._summarize_agent_logits(
        torch.tensor([[5.0, 4.0, 3.0, 2.0]]), torch.tensor([0])
    )

    assert trace["head_argmax_id"] == trace["predicted_token_id"] == 0
    assert trace["head_argmax_logit"] == 5.0
    assert trace["head_argmax_equal_count"] == 1
    assert trace["watched"]["bos"]["rank"] == 2
    assert [item["token_id"] for item in trace["top_k"]] == [0, 1, 2]
    assert all(not hasattr(value, "detach") for value in trace.values())

    with pytest.raises(RuntimeError, match="argmax does not match"):
        interface._summarize_agent_logits(
            torch.tensor([[5.0, 4.0, 3.0, 2.0]]), torch.tensor([1])
        )
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        interface._summarize_agent_logits(
            torch.tensor([[float("nan"), 4.0, 3.0, 2.0]]), torch.tensor([0])
        )
    with pytest.raises(RuntimeError, match="text_logits custom output is missing"):
        interface._attach_agent_logit_trace(
            {"predicted_token": torch.tensor([0])},
            SimpleNamespace(custom_outputs={"function_tokens": torch.tensor([0])}),
        )

    synthetic = {"predicted_token": torch.tensor([0])}
    interface._attach_agent_logit_trace(
        synthetic,
        SimpleNamespace(
            custom_outputs={
                "function_tokens": torch.tensor([0]),
                "voicechat_pad_pair_buffered": True,
            },
            total_tokens=7,
        ),
    )
    assert synthetic["agent_logit_trace"] == {
        "no_model_position": True,
        "scheduler_decision": "pad_pair_buffered",
        "synthetic_token_id": 0,
        "total_tokens": 7,
    }
    with pytest.raises(RuntimeError, match="contradictory"):
        interface._attach_agent_logit_trace(
            {"predicted_token": torch.tensor([0])},
            SimpleNamespace(
                custom_outputs={
                    "function_tokens": torch.tensor([0]),
                    "text_logits": torch.tensor([[1.0, 0.0]]),
                    "voicechat_pad_pair_buffered": True,
                },
                total_tokens=7,
            ),
        )
    with pytest.raises(RuntimeError, match="contradictory"):
        interface._attach_agent_logit_trace(
            {"predicted_token": torch.tensor([0])},
            SimpleNamespace(
                custom_outputs={
                    "function_tokens": torch.tensor([0]),
                    "voicechat_pad_pair_buffered": False,
                },
                total_tokens=7,
            ),
        )
    with pytest.raises(RuntimeError, match="requires synthetic PAD"):
        interface._attach_agent_logit_trace(
            {"predicted_token": torch.tensor([1])},
            SimpleNamespace(
                custom_outputs={
                    "function_tokens": torch.tensor([0]),
                    "voicechat_pad_pair_buffered": True,
                },
                total_tokens=7,
            ),
        )
    with pytest.raises(RuntimeError, match="requires synthetic PAD"):
        interface._attach_agent_logit_trace(
            {"predicted_token": torch.tensor([0])},
            SimpleNamespace(
                custom_outputs={
                    "function_tokens": torch.tensor([1]),
                    "voicechat_pad_pair_buffered": True,
                },
                total_tokens=7,
            ),
        )


def test_installed_agent_logit_trace_rejects_empty_generation() -> None:
    import torch

    model_factory = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.model_factory",
        reason="NVIDIA Speech is supplied by the runtime container",
    )

    class EmptyEngine:
        async def generate_next_token(self, *_args, **_kwargs):
            return None

    interface = model_factory.VllmLLMModel.__new__(model_factory.VllmLLMModel)
    interface.engine = EmptyEngine()
    interface._text_pad_id = 0
    with pytest.raises(RuntimeError, match="no token was generated"):
        asyncio.run(
            interface._process_inputs_to_outputs(
                torch.zeros(1, 1, 4),
                "request-1",
                capture_agent_logit_trace=True,
            )
        )


def test_installed_wrapper_accepts_bounded_buffered_tail_shape() -> None:
    wrapper_module = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper",
        reason="NVIDIA Speech is supplied by the runtime container",
    )
    wrapper = wrapper_module.NemotronVoicechatInferenceWrapper.__new__(
        wrapper_module.NemotronVoicechatInferenceWrapper
    )
    wrapper.model = SimpleNamespace(stt_model=SimpleNamespace(text_pad_id=0))
    wrapper.tokenizer = SimpleNamespace(ids_to_text=lambda token_ids: str(token_ids[0]))

    real = {
        "predicted_token_id": 31,
        "watched": {},
        "top_k": [{"token_id": 31, "logit": 1.0}],
    }
    buffered = {
        "no_model_position": True,
        "scheduler_decision": "pad_pair_buffered",
        "synthetic_token_id": 0,
        "total_tokens": 7,
    }
    rows = [
        wrapper._trace_agent_topk(real, 31),
        wrapper._trace_agent_topk(real, 31),
        wrapper._trace_agent_topk(real, 31),
        *[wrapper._trace_agent_topk(buffered, 0) for _ in range(6)],
    ]

    assert len(rows) == 9
    assert all(row is not None for row in rows)
    assert all(row["no_model_position"] is True for row in rows[3:])
    with pytest.raises(RuntimeError, match="recorded raw PAD"):
        wrapper._trace_agent_topk(buffered, 1)
    with pytest.raises(RuntimeError, match="recorded raw PAD"):
        wrapper._trace_agent_topk({**buffered, "unexpected": True}, 0)


@pytest.mark.parametrize("value", ["-1", "21", "not-an-integer"])
def test_installed_agent_logit_trace_rejects_invalid_top_k(value: str) -> None:
    model_factory = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.model_factory",
        reason="NVIDIA Speech is supplied by the runtime container",
    )

    with pytest.raises(ValueError, match="integer from 0 through 20"):
        model_factory.VllmLLMModel._parse_agent_logit_trace_topk(value)


def test_installed_agent_logit_trace_rejects_production_artifact(tmp_path) -> None:
    model_factory = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.model_factory",
        reason="NVIDIA Speech is supplied by the runtime container",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"custom_outputs": ["function_tokens", "function_logits"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="weight-identical diagnostic"):
        model_factory.VllmLLMModel._validate_agent_logit_trace_artifact(
            str(tmp_path), 20
        )

    config.write_text(
        json.dumps(
            {
                "custom_outputs": [
                    "text_logits",
                    "function_tokens",
                    "function_logits",
                ]
            }
        ),
        encoding="utf-8",
    )
    model_factory.VllmLLMModel._validate_agent_logit_trace_artifact(
        str(tmp_path), 20
    )


@pytest.mark.parametrize("value", ["-1", "21", "not-an-integer"])
def test_installed_function_logit_trace_rejects_invalid_top_k(value: str) -> None:
    model_factory = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.model_factory",
        reason="NVIDIA Speech is supplied by the runtime container",
    )

    with pytest.raises(ValueError, match="integer from 0 through 20"):
        model_factory.VllmLLMModel._parse_function_logit_trace_topk(value)


def test_installed_function_logit_trace_is_disabled_for_eartts() -> None:
    model_factory = pytest.importorskip(
        "nemo.collections.speechlm2.inference.model_wrappers.model_factory",
        reason="NVIDIA Speech is supplied by the runtime container",
    )

    assert model_factory.VllmLLMModel._parse_function_logit_trace_topk(
        "10", model_type="eartts"
    ) == 0
    assert model_factory.VllmLLMModel._parse_function_logit_trace_topk(
        "10", model_type="llm"
    ) == 10
    with pytest.raises(ValueError, match="unsupported"):
        model_factory.VllmLLMModel._parse_function_logit_trace_topk(
            "10", model_type="unknown"
        )
