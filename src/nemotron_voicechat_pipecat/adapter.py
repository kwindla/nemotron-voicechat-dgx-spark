"""Universal-context adapter for Voicechat realtime."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage


class VoicechatInvocationParams(TypedDict):
    """The only context fields accepted during realtime session setup."""

    system_instruction: str | None
    tools: list[dict[str, Any]]


class NemotronVoicechatLLMAdapter(BaseLLMAdapter[VoicechatInvocationParams]):
    """Maps Pipecat tool schemas and initial instructions to Voicechat strict-v3."""

    @property
    def id_for_llm_specific_messages(self) -> str:
        return "nvidia-nemotron-voicechat"

    @staticmethod
    def _tool(function: FunctionSchema) -> dict[str, Any]:
        return {
            "type": "function",
            "name": function.name,
            "description": function.description,
            "parameters": {
                "type": "object",
                "properties": function.properties,
                "required": function.required,
            },
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[dict[str, Any]]:
        tools = [self._tool(function) for function in tools_schema.standard_tools]
        if tools_schema.custom_tools:
            tools.extend(tools_schema.custom_tools.get(AdapterType.OPENAI, []))
        return tools

    def get_llm_invocation_params(
        self,
        context: LLMContext,
        *,
        system_instruction: str | None = None,
    ) -> VoicechatInvocationParams:
        messages = self.get_messages(context)
        instruction_parts: list[str] = []
        if system_instruction:
            instruction_parts.append(system_instruction)

        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                raise TypeError("Voicechat does not accept provider-specific history")
            role = message.get("role")
            if role not in ("system", "developer"):
                raise ValueError(
                    "Voicechat sessions start with live audio; ordinary conversation "
                    "history cannot be replayed"
                )
            content = message.get("content")
            if isinstance(content, str):
                instruction_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        instruction_parts.append(item["text"])

        tools = self.from_standard_tools(context.tools)
        if tools is None or not isinstance(tools, list):
            tools = []
        return {
            "system_instruction": "\n\n".join(part for part in instruction_parts if part) or None,
            "tools": cast(list[dict[str, Any]], tools),
        }

    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.get_messages(context, truncate_large_values=True))
