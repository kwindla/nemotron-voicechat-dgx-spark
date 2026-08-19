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
from zoneinfo import ZoneInfo, available_timezones

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
from .playout_transport import (
    attach_underflow_counter,
    install_underflow_telemetry,
    install_warm_resampler,
    output_chunks_10ms,
)
from .text_input import VoicechatTypedInputRouter
from .tool_results import VoicechatLLMContext, VoicechatToolResult

SYSTEM_INSTRUCTION = """You are a helpful voice assistant in a live conversation.
Keep answers concise, natural, and easy to understand when spoken aloud.
Use a tool only when the user's request matches that tool's purpose. Never use an
available tool as a substitute for an unrelated or unavailable capability. After a
tool returns, state the MOST RECENT result accurately and continue the conversation
normally. Never restate an older tool result after a newer result supersedes it."""
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


DEFAULT_TIMEZONE = "UTC"

# Spoken requests name regions, not IANA keys. Map the common spoken forms;
# anything else falls through to a case-insensitive IANA lookup.
TIMEZONE_ALIASES = {
    "pacific": "America/Los_Angeles",
    "pacific time": "America/Los_Angeles",
    "us pacific": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "mountain": "America/Denver",
    "mountain time": "America/Denver",
    "mt": "America/Denver",
    "central": "America/Chicago",
    "central time": "America/Chicago",
    "ct": "America/Chicago",
    "eastern": "America/New_York",
    "eastern time": "America/New_York",
    "et": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "utc": "UTC",
    "gmt": "UTC",
    "zulu": "UTC",
    "uk": "Europe/London",
    "london": "Europe/London",
    "britain": "Europe/London",
    "india": "Asia/Kolkata",
    "japan": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo",
}

# Process state: the time zone the bot reports times in. Set by the
# set_timezone tool and retained for the life of the process, so a later
# get_current_time call observes an earlier set_timezone call.
_reporting_timezone = DEFAULT_TIMEZONE


def reporting_timezone() -> str:
    """Return the process-wide reporting time zone."""
    return _reporting_timezone


def reset_reporting_timezone() -> None:
    """Restore the default reporting time zone (test/support hook)."""
    global _reporting_timezone
    _reporting_timezone = DEFAULT_TIMEZONE


def resolve_timezone(requested: str) -> str | None:
    """Resolve a spoken or IANA time-zone name, or None when unknown."""
    if not isinstance(requested, str):
        return None
    cleaned = requested.strip()
    if not cleaned:
        return None
    alias = TIMEZONE_ALIASES.get(cleaned.casefold())
    if alias is not None:
        return alias
    available = available_timezones()
    if cleaned in available:
        return cleaned
    folded = {name.casefold(): name for name in available}
    return folded.get(cleaned.casefold())


def spoken_time(now: datetime) -> str:
    """Render a clock time the way a voice assistant should say it."""
    return f"{now.strftime('%I:%M %p').lstrip('0')} {now.tzname()}"


async def get_current_time(params: FunctionCallParams):
    """Get the current time in the process-wide reporting time zone."""
    zone = _reporting_timezone
    now = datetime.now(ZoneInfo(zone))
    spoken = spoken_time(now)
    await params.result_callback(
        VoicechatToolResult(
            output={
                "timezone": zone,
                "iso_local": now.isoformat(),
                "iso_utc": now.astimezone(UTC).isoformat(),
                "spoken": spoken,
            },
            model_output=f"It is currently {spoken}.",
        )
    )


async def set_timezone(params: FunctionCallParams):
    """Set the time zone used by later get_current_time calls."""
    global _reporting_timezone
    requested = (getattr(params, "arguments", None) or {}).get("timezone")
    resolved = resolve_timezone(requested)
    if resolved is None:
        # Report the failure as an ordinary result so the model can ask the
        # user to clarify instead of the turn failing.
        await params.result_callback(
            VoicechatToolResult(
                output={
                    "ok": False,
                    "requested": requested,
                    "timezone": _reporting_timezone,
                },
                model_output=(
                    f"I could not set the time zone to {requested!r}; it is not a time "
                    f"zone I recognize. The time zone is still {_reporting_timezone}."
                ),
            )
        )
        return
    _reporting_timezone = resolved
    now = datetime.now(ZoneInfo(resolved))
    await params.result_callback(
        VoicechatToolResult(
            output={
                "ok": True,
                "requested": requested,
                "timezone": resolved,
                "spoken": spoken_time(now),
            },
            model_output=f"The time zone is now {resolved}.",
        )
    )


def create_tools() -> ToolsSchema:
    """Return the bot's advertised tools and their Pipecat handlers."""
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="get_current_time",
                description=(
                    "Get the current clock time in the time zone this conversation "
                    "is using."
                ),
                properties={},
                required=[],
                handler=get_current_time,
            ),
            FunctionSchema(
                name="set_timezone",
                description=(
                    "Set the time zone used when reporting the current time. Accepts "
                    "an IANA name such as America/Los_Angeles or a common spoken name "
                    "such as Pacific."
                ),
                properties={
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA time zone name (America/Los_Angeles) or common "
                            "spoken name (Pacific, Eastern, UTC, Tokyo)."
                        ),
                    }
                },
                required=["timezone"],
                handler=set_timezone,
            ),
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
    # Reclaim playout lead the transport would otherwise discard, and count
    # auto-silence emissions when telemetry is enabled. Both must be installed
    # before the transport builds its resampler and output track.
    install_warm_resampler()
    install_underflow_telemetry()
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
            # How much audio BaseOutputTransport hands the WebRTC track at a
            # time. Smaller values strand less generated audio in the output
            # byte buffer, but the track is fed by a separate realtime clock,
            # so this is also the slack absorbing jitter between the two
            # clocks. Tunable so the tradeoff can be measured; see
            # playout_transport and docs/reviews/audio-crackle-root-cause-r1.md.
            audio_out_10ms_chunks=output_chunks_10ms(),
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
    underflow_counter = None

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info(
            "SmallWebRTC client connected; Voicechat backend={} typed_input=server-owned",
            service.base_url,
        )
        # Hold the counter now: the transport clears its output-track
        # reference before firing on_client_disconnected.
        nonlocal underflow_counter
        underflow_counter = attach_underflow_counter(transport)
        # Push the initial universal context before releasing the service's
        # bounded audio pre-roll. Voicechat itself decides when to respond.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("SmallWebRTC client disconnected")
        if underflow_counter is not None:
            logger.info("voicechat.playout.track_underflow {}", underflow_counter.summary())
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
