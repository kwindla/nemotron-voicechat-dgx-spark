"""Universal-context and tool conversion tests."""

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

from nemotron_voicechat_pipecat.adapter import NemotronVoicechatLLMAdapter


def test_adapter_maps_instruction_and_function_schema():
    adapter = NemotronVoicechatLLMAdapter()
    tool = FunctionSchema(
        name="get_weather",
        description="Get weather",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    context = LLMContext(
        messages=[{"role": "system", "content": "Speak briefly."}],
        tools=ToolsSchema(standard_tools=[tool]),
    )

    params = adapter.get_llm_invocation_params(context)

    assert params["system_instruction"] == "Speak briefly."
    assert params["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_adapter_refuses_conversation_replay():
    context = LLMContext(messages=[{"role": "user", "content": "old turn"}])
    with pytest.raises(ValueError, match="cannot be replayed"):
        NemotronVoicechatLLMAdapter().get_llm_invocation_params(context)
