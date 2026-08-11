"""Application wiring tests for the SmallWebRTC bot."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from nemotron_voicechat_pipecat import NemotronVoicechatLLMService
from nemotron_voicechat_pipecat import demo as bot_module
from nemotron_voicechat_pipecat.aggregators import (
    VoicechatAssistantAggregator,
    VoicechatUserAggregator,
    create_voicechat_context_aggregators,
)
from nemotron_voicechat_pipecat.text_input import VoicechatTypedInputRouter
from nemotron_voicechat_pipecat.tool_results import (
    VoicechatLLMContext,
    VoicechatToolResult,
)


@pytest.mark.asyncio
async def test_browser_bot_ready_waits_for_live_model_input(monkeypatch):
    gate = asyncio.Event()
    observed = []

    class Service:
        async def wait_for_live_input_ready(self):
            observed.append("wait")
            await gate.wait()

    async def publish_ready(_processor, about=None):
        observed.append(("published", about))

    monkeypatch.setattr(bot_module.RTVIProcessor, "set_bot_ready", publish_ready)
    processor = bot_module.VoicechatReadyRTVIProcessor(Service())
    task = asyncio.create_task(processor.set_bot_ready(about={"source": "test"}))
    await asyncio.sleep(0)
    assert observed == ["wait"]
    gate.set()
    await task
    assert observed == ["wait", ("published", {"source": "test"})]


@pytest.mark.asyncio
async def test_browser_client_ready_serializes_later_messages_behind_model(monkeypatch):
    gate = asyncio.Event()
    observed = []

    class Service:
        async def wait_for_live_input_ready(self):
            observed.append("wait")
            await gate.wait()

    async def mark_client_ready(_processor):
        observed.append("client-ready")

    monkeypatch.setattr(bot_module.RTVIProcessor, "set_client_ready", mark_client_ready)
    processor = bot_module.VoicechatReadyRTVIProcessor(Service())
    task = asyncio.create_task(processor.set_client_ready())
    await asyncio.sleep(0)

    assert observed == ["wait"]
    gate.set()
    await task
    assert observed == ["wait", "client-ready"]


@pytest.mark.asyncio
async def test_browser_client_ready_timeout_is_fatal_and_closes_model(monkeypatch):
    observed = []

    class Service:
        async def wait_for_live_input_ready(self):
            raise TimeoutError

        def live_input_readiness_status(self):
            return {
                "session_ready": True,
                "context_ready": False,
                "audio_send_task": True,
            }

        async def abort_live_input_startup(self):
            observed.append("aborted")

    async def record_error(frame):
        observed.append(frame)

    processor = bot_module.VoicechatReadyRTVIProcessor(Service())
    monkeypatch.setattr(processor, "_send_error_frame", record_error)

    await processor.set_client_ready()

    assert isinstance(observed[0], ErrorFrame)
    assert observed[0].fatal is True
    assert "session_ready=True" in observed[0].error
    assert "context_ready=False" in observed[0].error
    assert observed[1] == "aborted"


@pytest.mark.asyncio
async def test_tool_handler_returns_terminal_utc_result():
    results = []

    async def result_callback(result, **_kwargs):
        results.append(result)

    await bot_module.get_current_utc_time(SimpleNamespace(result_callback=result_callback))

    assert len(results) == 1
    assert isinstance(results[0], VoicechatToolResult)
    assert results[0].output["timezone"] == "UTC"
    assert results[0].output["spoken"].endswith(" UTC")
    assert results[0].model_output.startswith("It is currently ")
    assert results[0].model_output.endswith(" UTC.")


@pytest.mark.asyncio
async def test_concise_tool_result_retains_full_context_and_sidecar(monkeypatch):
    context = VoicechatLLMContext(messages=[])
    _user, assistant = create_voicechat_context_aggregators(context)
    call = FunctionCallFromLLM(
        function_name="get_current_utc_time",
        tool_call_id="call_concise",
        arguments={},
        context=context,
    )
    await assistant._handle_function_calls_started(FunctionCallsStartedFrame([call]))
    await assistant._handle_function_call_in_progress(
        FunctionCallInProgressFrame(
            function_name=call.function_name,
            tool_call_id=call.tool_call_id,
            arguments={},
            cancel_on_interruption=True,
        )
    )

    async def record_context_push(*_args, **_kwargs):
        return None

    monkeypatch.setattr(assistant, "push_context_frame", record_context_push)
    await assistant._handle_function_call_result(
        FunctionCallResultFrame(
            function_name=call.function_name,
            tool_call_id=call.tool_call_id,
            arguments={},
            result=VoicechatToolResult(
                output={"timezone": "UTC", "iso_utc": "full-value"},
                model_output="It is currently four o'clock.",
            ),
        )
    )

    assert context.get_messages()[-1]["content"] == (
        '{"timezone": "UTC", "iso_utc": "full-value"}'
    )
    assert context.voicechat_model_output("call_concise") == (
        "It is currently four o'clock."
    )


@pytest.mark.asyncio
async def test_tool_result_reaches_provider_before_provider_owned_eou(monkeypatch):
    """A function call must not wait on an EOU that needs its own result."""
    context = LLMContext(messages=[])
    _user, assistant = create_voicechat_context_aggregators(context)
    call = FunctionCallFromLLM(
        function_name="get_current_utc_time",
        tool_call_id="call_time",
        arguments={},
        context=context,
    )
    await assistant._handle_function_calls_started(FunctionCallsStartedFrame([call]))
    await assistant._handle_function_call_in_progress(
        FunctionCallInProgressFrame(
            function_name=call.function_name,
            tool_call_id=call.tool_call_id,
            arguments={},
            cancel_on_interruption=True,
        )
    )
    assistant._user_speaking = True
    pushes = []

    async def record_context_push(direction=FrameDirection.DOWNSTREAM):
        pushes.append(direction)

    monkeypatch.setattr(assistant, "push_context_frame", record_context_push)
    await assistant._handle_function_call_result(
        FunctionCallResultFrame(
            function_name=call.function_name,
            tool_call_id=call.tool_call_id,
            arguments={},
            result={"timezone": "UTC", "spoken": "01:23 UTC"},
        )
    )

    assert pushes == [FrameDirection.UPSTREAM]
    assert assistant._user_speaking is True
    assert context.get_messages()[-1] == {
        "role": "tool",
        "tool_call_id": "call_time",
        "content": '{"timezone": "UTC", "spoken": "01:23 UTC"}',
    }


@pytest.mark.asyncio
async def test_pipeline_uses_smallwebrtc_and_universal_realtime_aggregators(
    monkeypatch,
):
    monkeypatch.setenv("NEMOTRON_VOICECHAT_WS_URL", "ws://model.test/v1/realtime")
    connection = SmallWebRTCConnection()
    args = SmallWebRTCRunnerArguments(webrtc_connection=connection)
    transport = bot_module.create_transport(args)
    try:
        pipeline, context, service = bot_module.create_pipeline(transport)
        names = [type(processor).__name__ for processor in pipeline.processors]
        user_aggregator = next(
            processor
            for processor in pipeline.processors
            if isinstance(processor, VoicechatUserAggregator)
        )

        assert isinstance(transport, SmallWebRTCTransport)
        assert isinstance(service, NemotronVoicechatLLMService)
        assert service.base_url == "ws://model.test/v1/realtime"
        assert names.index("VoicechatTypedInputRouter") < names.index("VoicechatUserAggregator")
        assert names.index("VoicechatUserAggregator") < names.index("NemotronVoicechatLLMService")
        assert any(
            isinstance(processor, VoicechatTypedInputRouter) for processor in pipeline.processors
        )
        assert any(
            isinstance(processor, VoicechatUserAggregator) for processor in pipeline.processors
        )
        assert isinstance(user_aggregator._params.vad_analyzer, SileroVADAnalyzer)
        assert user_aggregator._params.vad_analyzer.params.start_secs == 0.2
        assert user_aggregator._params.vad_analyzer.params.stop_secs == 0.2
        assert user_aggregator._params.user_turn_stop_timeout == 5.0
        strategies = user_aggregator._params.user_turn_strategies
        assert isinstance(strategies.start[0], VADUserTurnStartStrategy)
        assert isinstance(strategies.stop[0], TurnAnalyzerUserTurnStopStrategy)
        assert isinstance(strategies.stop[0]._turn_analyzer, LocalSmartTurnAnalyzerV3)
        assert strategies.stop[0].wait_for_transcript is False
        assert names.index("NemotronVoicechatLLMService") < names.index(
            "VoicechatAssistantAggregator"
        )
        assert any(
            isinstance(processor, VoicechatAssistantAggregator) for processor in pipeline.processors
        )
        assert all("VADProcessor" not in name for name in names)
        assert [tool.name for tool in context.tools.standard_tools] == ["get_current_utc_time"]
    finally:
        await connection._pc.close()


@pytest.mark.asyncio
async def test_bot_rejects_non_smallwebrtc_runner_arguments():
    with pytest.raises(TypeError, match="only Pipecat SmallWebRTC"):
        await bot_module.bot(RunnerArguments())
