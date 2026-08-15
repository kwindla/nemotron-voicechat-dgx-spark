"""Explicit full and model-visible results for Voicechat tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipecat.processors.aggregators.llm_context import LLMContext


@dataclass(frozen=True)
class VoicechatToolResult:
    """Carry full application data plus concise, client-authored model context.

    ``output`` is retained as the canonical Pipecat/tool result. ``model_output``
    is the only value injected into Voicechat when the peer negotiated that
    optional protocol capability. The runtime never derives one from the other.
    """

    output: Any
    model_output: str


class VoicechatLLMContext(LLMContext):
    """LLM context with a call-correlated sidecar for concise tool results."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._voicechat_model_outputs: dict[str, str] = {}

    def set_voicechat_model_output(self, call_id: str, model_output: str) -> None:
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Voicechat model output requires a call ID")
        if not isinstance(model_output, str) or not model_output.strip():
            raise ValueError("Voicechat model output must be a non-empty string")
        if call_id in self._voicechat_model_outputs:
            raise ValueError(f"duplicate Voicechat model output for call_id={call_id}")
        self._voicechat_model_outputs[call_id] = model_output

    def voicechat_model_output(self, call_id: str) -> str | None:
        return self._voicechat_model_outputs.get(call_id)

    def discard_voicechat_model_output(self, call_id: str) -> None:
        self._voicechat_model_outputs.pop(call_id, None)
