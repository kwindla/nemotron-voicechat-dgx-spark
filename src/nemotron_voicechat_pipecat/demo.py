"""SmallWebRTC Pipecat bot for the strict Nemotron Voicechat service.

Run with::

    uv run python bot.py -t webrtc --host 0.0.0.0 --port 7860

The pinned Pipecat runner serves its browser client at ``/client/`` and owns
WebRTC signaling at ``/api/offer``.  The browser never connects to the model
WebSocket directly.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

# NLTK 3.10 installs a guard against dependencies resolved from the current
# working directory. A project-local uv environment is below that directory,
# so its lazy ``regex`` import is rejected when the runner starts. Importing
# the explicitly pinned dependency before Pipecat/NLTK avoids the false positive.
import regex as _regex  # noqa: F401

_PROCESS_WS_URL = os.getenv("NEMOTRON_VOICECHAT_WS_URL")

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import ErrorFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from .aggregators import create_voicechat_context_aggregators
from .llm import NemotronVoicechatLLMService
from .text_input import VoicechatTypedInputRouter
from .tool_results import VoicechatLLMContext, VoicechatToolResult

SYSTEM_INSTRUCTION = """You are a helpful voice assistant in a live conversation.
Keep answers concise, natural, and easy to understand when spoken aloud.
Use a tool only when the user's request matches that tool's purpose. Never use an
available tool as a substitute for an unrelated or unavailable capability. After a
tool returns, state its result accurately and continue the conversation normally."""
STEP4C_FIXTURE_ENV = "NEMOTRON_VOICECHAT_QUALIFICATION_FIXTURE"
STEP4C_FIXTURE = "step4c-l1"
STEP4C_SYSTEM_INSTRUCTION = (
    "Qualification latency fixture: when the user supplies a script, speak the "
    "supplied script aloud completely and exactly, then stop."
)


class VoicechatReadyRTVIProcessor(RTVIProcessor):
    """Publish browser bot-ready only after the model session can accept input."""

    def __init__(self, service: NemotronVoicechatLLMService):
        super().__init__()
        self._voicechat_service = service

    async def set_client_ready(self):
        """Serialize every later RTVI message behind model input readiness."""
        try:
            await self._voicechat_service.wait_for_live_input_ready()
        except TimeoutError:
            readiness = self._voicechat_service.live_input_readiness_status()
            message = "Voicechat model input did not become ready: " + " ".join(
                f"{key}={value}" for key, value in readiness.items()
            )
            logger.error(message)
            await self._send_error_frame(ErrorFrame(error=message, fatal=True))
            await self._voicechat_service.abort_live_input_startup()
            return
        await super().set_client_ready()

    async def set_bot_ready(self, about=None):
        await self._voicechat_service.wait_for_live_input_ready()
        await super().set_bot_ready(about=about)


async def get_current_utc_time(params: FunctionCallParams):
    """Get the current time in UTC."""
    now = datetime.now(UTC)
    await params.result_callback(
        VoicechatToolResult(
            output={
                "timezone": "UTC",
                "iso_utc": now.isoformat(),
                "spoken": now.strftime("%H:%M UTC"),
            },
            model_output=f"It is currently {now.strftime('%H:%M UTC')}.",
        )
    )


def create_tools() -> ToolsSchema:
    """Return the bot's advertised tools and their Pipecat handlers."""
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="get_current_utc_time",
                description="Get the current clock time in UTC.",
                properties={},
                required=[],
                handler=get_current_utc_time,
            )
        ]
    )


def qualification_settings() -> tuple[str, ToolsSchema | list[object]]:
    """Freeze the optional Step 4c browser configuration at process startup."""

    fixture = os.getenv(STEP4C_FIXTURE_ENV, "").strip()
    if not fixture:
        return SYSTEM_INSTRUCTION, create_tools()
    if fixture != STEP4C_FIXTURE:
        raise RuntimeError(f"unsupported browser qualification fixture: {fixture}")
    return STEP4C_SYSTEM_INSTRUCTION, []


def create_transport(runner_args: SmallWebRTCRunnerArguments) -> SmallWebRTCTransport:
    """Create the required SmallWebRTC media transport."""
    return SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16_000,
            # WebRTC output uses an integral 10 ms cadence; the transport
            # resamples Voicechat's native 22.05 kHz output to 24 kHz.
            audio_out_sample_rate=24_000,
            audio_in_channels=1,
            audio_out_channels=1,
        ),
    )


def create_pipeline(
    transport: BaseTransport,
):
    """Build the server-turn-driven realtime pipeline and shared context."""
    system_instruction, tools = qualification_settings()
    context = VoicechatLLMContext(messages=[], tools=tools)
    user_aggregator, assistant_aggregator = create_voicechat_context_aggregators(context)
    typed_input = VoicechatTypedInputRouter()
    service = NemotronVoicechatLLMService(
        base_url=_PROCESS_WS_URL
        or os.getenv("NEMOTRON_VOICECHAT_WS_URL", "ws://127.0.0.1:8786/v1/realtime"),
        system_instruction=system_instruction,
        tools=tools,
    )
    pipeline = Pipeline(
        [
            transport.input(),
            typed_input,
            user_aggregator,
            service,
            transport.output(),
            assistant_aggregator,
        ]
    )
    return pipeline, context, service


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run one browser-owned Pipecat worker."""
    pipeline, _context, service = create_pipeline(transport)
    worker = PipelineWorker(
        pipeline,
        # Browser transcript/event delivery is part of the qualified contract;
        # do not rely on Pipecat's current default remaining enabled.
        enable_rtvi=True,
        params=PipelineParams(
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=24_000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        rtvi_processor=VoicechatReadyRTVIProcessor(service),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info(
            "SmallWebRTC client connected; Voicechat backend={} typed_input=server-owned",
            service.base_url,
        )
        # Push the initial universal context before releasing the service's
        # bounded audio pre-roll. Voicechat itself decides when to respond.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("SmallWebRTC client disconnected")
        await worker.cancel()

    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint,
        handle_sigterm=runner_args.handle_sigterm,
    )
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Pipecat development-runner entry point."""
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        raise TypeError("This bot supports only Pipecat SmallWebRTC transport")
    await run_bot(create_transport(runner_args), runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
