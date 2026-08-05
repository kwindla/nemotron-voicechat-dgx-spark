"""Component tests for the strict-v3 typed-input router."""

from __future__ import annotations

import pytest
import regex as _regex  # noqa: F401 - preempt NLTK's cwd import guard
from pipecat.frames.frames import (
    ErrorFrame,
    InputTextRawFrame,
    LLMConfigureOutputFrame,
    LLMMessagesAppendFrame,
)
from pipecat.tests.utils import run_test

from nemotron_voicechat_pipecat.text_input import (
    VoicechatTypedInputFrame,
    VoicechatTypedInputRouter,
)


def typed_frame(text: str, *, run_llm: bool = True) -> LLMMessagesAppendFrame:
    return LLMMessagesAppendFrame(
        messages=[{"role": "user", "content": text}],
        run_llm=run_llm,
    )


@pytest.mark.asyncio
async def test_byte_exact_text_becomes_one_server_request_without_context_append():
    router = VoicechatTypedInputRouter()
    text = "  Remember sapphire.  "

    down, up = await run_test(router, frames_to_send=[typed_frame(text)])

    requests = [item for item in down if isinstance(item, VoicechatTypedInputFrame)]
    assert len(requests) == 1
    assert requests[0].text == text
    assert len(requests[0].job_id) == 32
    assert not [item for item in down if isinstance(item, LLMMessagesAppendFrame)]
    assert not [item for item in up if isinstance(item, ErrorFrame)]


@pytest.mark.asyncio
async def test_raw_text_uses_the_same_typed_request_path():
    router = VoicechatTypedInputRouter()
    down, _ = await run_test(router, frames_to_send=[InputTextRawFrame(text="Hello")])
    requests = [item for item in down if isinstance(item, VoicechatTypedInputFrame)]
    assert [item.text for item in requests] == ["Hello"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefix", "frame", "message"),
    [
        (
            [LLMConfigureOutputFrame(skip_tts=True)],
            typed_frame("Text only."),
            "audio_response=true",
        ),
        ([], typed_frame("Later.", run_llm=False), "run_immediately=true"),
        (
            [],
            LLMMessagesAppendFrame(
                messages=[
                    {"role": "user", "content": "One"},
                    {"role": "user", "content": "Two"},
                ],
                run_llm=True,
            ),
            "exactly one",
        ),
        ([], typed_frame(" "), "non-empty"),
    ],
)
async def test_invalid_typed_options_have_no_downstream_side_effect(prefix, frame, message):
    router = VoicechatTypedInputRouter()
    down, up = await run_test(router, frames_to_send=[*prefix, frame])
    assert not [item for item in down if isinstance(item, VoicechatTypedInputFrame)]
    errors = [item for item in up if isinstance(item, ErrorFrame)]
    assert len(errors) == 1
    assert message in errors[0].error


@pytest.mark.asyncio
async def test_text_limit_is_enforced_before_the_wire_boundary():
    router = VoicechatTypedInputRouter(max_text_chars=4)
    down, up = await run_test(router, frames_to_send=[typed_frame("12345")])
    assert not [item for item in down if isinstance(item, VoicechatTypedInputFrame)]
    assert "4-character" in next(item for item in up if isinstance(item, ErrorFrame)).error

