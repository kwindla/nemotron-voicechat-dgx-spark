"""Universal context aggregators adapted for provider-driven realtime tools."""

from __future__ import annotations

from dataclasses import replace

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import FunctionCallResultFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .tool_results import VoicechatLLMContext, VoicechatToolResult


class VoicechatUserAggregator(LLMUserAggregator):
    """Universal user aggregator for provider-owned realtime turns."""


class VoicechatAssistantAggregator(LLMAssistantAggregator):
    """Forward terminal tool results even before provider-owned EOU.

    Voicechat can request a function while its server-side speech turn remains
    open. The provider then waits for ``function_call_output`` before it can
    resume inference and emit ``speech_stopped``. Pipecat's universal assistant
    aggregator normally suppresses the result-triggered context push while the
    user-speaking flag is set, which creates a circular wait for this ordering.

    Function result frames are uninterruptible and already carry their own
    ``run_llm`` policy. Temporarily lifting only the user-speaking guard lets the
    universal implementation retain all result, grouping, bot-speaking, and
    context semantics while delivering the result to the realtime provider.
    """

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame):
        if isinstance(frame.result, VoicechatToolResult):
            if not isinstance(self._context, VoicechatLLMContext):
                raise TypeError(
                    "VoicechatToolResult requires a VoicechatLLMContext sidecar"
                )
            self._context.set_voicechat_model_output(
                frame.tool_call_id, frame.result.model_output
            )
            frame = replace(frame, result=frame.result.output)
        provider_turn_open = self._user_speaking
        if provider_turn_open:
            self._user_speaking = False
        try:
            await super()._handle_function_call_result(frame)
        finally:
            if provider_turn_open:
                self._user_speaking = True


def create_voicechat_context_aggregators(
    context: LLMContext,
):
    """Create the paired universal aggregators used by the Voicechat bot."""
    user = VoicechatUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    min_volume=0.6,
                    start_secs=0.2,
                    stop_secs=0.2,
                )
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(cpu_count=1),
                        wait_for_transcript=False,
                    )
                ],
            ),
            user_turn_stop_timeout=5.0,
        ),
    )
    assistant = VoicechatAssistantAggregator(
        context,
        _paired_user_aggregator=user,
    )
    return user, assistant
