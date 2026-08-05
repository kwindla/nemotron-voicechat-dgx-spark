"""Pipecat routing for server-owned Voicechat typed input."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    InputTextRawFrame,
    LLMConfigureOutputFrame,
    LLMMessagesAppendFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class VoicechatTypedInputFrame(Frame):
    """One validated byte-exact typed utterance for the model server."""

    text: str = ""
    job_id: str = ""


class VoicechatTypedInputRouter(FrameProcessor):
    """Convert RTVI text frames into strict-v3 typed-input requests.

    The model server is the half-duplex authority and acknowledges acceptance.
    Consequently this processor does not append the text to context itself.
    The service commits it exactly once only after ``input_text.accepted``.
    """

    def __init__(self, *, max_text_chars: int = 1_000, **kwargs):
        super().__init__(**kwargs)
        self._max_text_chars = max_text_chars
        self._skip_tts = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMConfigureOutputFrame):
            self._skip_tts = frame.skip_tts
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMMessagesAppendFrame):
            error = self._validate_append(frame)
            if error is not None:
                await self.push_error(error_msg=error)
                return
            await self.push_frame(
                VoicechatTypedInputFrame(
                    text=frame.messages[0]["content"],
                    job_id=uuid.uuid4().hex,
                )
            )
            return
        if isinstance(frame, InputTextRawFrame):
            text = frame.text
            error = self._validate_text(text)
            if error is not None:
                await self.push_error(error_msg=error)
                return
            await self.push_frame(VoicechatTypedInputFrame(text=text, job_id=uuid.uuid4().hex))
            return
        await self.push_frame(frame, direction)

    def _validate_append(self, frame: LLMMessagesAppendFrame) -> str | None:
        if self._skip_tts:
            return "Voicechat typed input requires audio_response=true"
        if frame.run_llm is not True:
            return "Voicechat typed input requires run_immediately=true"
        if len(frame.messages) != 1:
            return "Voicechat typed input requires exactly one user message"
        message = frame.messages[0]
        if not isinstance(message, dict) or message.get("role") != "user":
            return "Voicechat typed input requires one user-role message"
        return self._validate_text(message.get("content"))

    def _validate_text(self, text) -> str | None:
        if self._skip_tts:
            return "Voicechat typed input requires audio_response=true"
        if not isinstance(text, str) or not text.strip():
            return "Voicechat typed input requires non-empty string content"
        if len(text) > self._max_text_chars:
            return f"Voicechat typed input exceeds the {self._max_text_chars}-character limit"
        return None

