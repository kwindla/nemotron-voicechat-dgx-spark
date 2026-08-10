"""Application wiring tests for the SmallWebRTC bot."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
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


@pytest.mark.asyncio
async def test_tool_handler_returns_terminal_utc_result():
    results = []

    async def result_callback(result, **_kwargs):
        results.append(result)

    await bot_module.get_current_utc_time(SimpleNamespace(result_callback=result_callback))

    assert len(results) == 1
    assert results[0]["timezone"] == "UTC"
    assert results[0]["spoken"].endswith(" UTC")


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
