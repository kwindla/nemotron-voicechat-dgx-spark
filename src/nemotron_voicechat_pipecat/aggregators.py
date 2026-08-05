"""Universal context aggregators adapted for provider-driven realtime tools."""

from __future__ import annotations

from pipecat.frames.frames import FunctionCallResultFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)


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
    user = VoicechatUserAggregator(context)
    assistant = VoicechatAssistantAggregator(
        context,
        _paired_user_aggregator=user,
    )
    return user, assistant
