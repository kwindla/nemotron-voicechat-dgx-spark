#!/usr/bin/env python3
"""Drive a paced typed-prompt soak through the native Pipecat service layer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path

# Import the pinned dependency before Pipecat/NLTK's project-path import guard.
import regex as _regex  # noqa: F401
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner
from step4c_contract import (
    DRAIN_SECONDS as STEP4C_DRAIN_SECONDS,
)
from step4c_contract import (
    DURATION_SECONDS as STEP4C_DURATION_SECONDS,
)
from step4c_contract import (
    L1_SHA256,
    L1_TEXT,
    QUALIFICATION_FIXTURE,
    SYSTEM_INSTRUCTION,
)
from step4c_contract import (
    WARMUP_SECONDS as STEP4C_WARMUP_SECONDS,
)

from nemotron_voicechat_pipecat.llm import NemotronVoicechatLLMService
from nemotron_voicechat_pipecat.text_input import VoicechatTypedInputFrame
from nemotron_voicechat_pipecat.tool_results import VoicechatLLMContext

FRAME_SECONDS = 0.08
INPUT_RATE = 16_000
PROMPTS = (
    "What is two plus three? Answer in one short sentence.",
    "Remember that the codeword is sapphire and acknowledge it briefly.",
    "What codeword did I ask you to remember?",
    "Say the word latency once.",
)


class QualificationService(NemotronVoicechatLLMService):
    """Expose the server terminal identity/status to the qualification sink."""

    step4_terminal_response_id: str | None = None
    step4_terminal_status: str | None = None

    async def _response_done(self, event: dict[str, object]):
        response_id = event.get("response_id")
        status = event.get("status")
        self.step4_terminal_response_id = response_id if isinstance(response_id, str) else None
        self.step4_terminal_status = status if isinstance(status, str) else "completed"
        await super()._response_done(event)


class CaptureSink(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.audio_frames = 0
        self.audio_bytes = 0
        self.responses_done = 0
        self.service: QualificationService | None = None
        self.defer_terminal_identity = False
        self.job_records: list[dict[str, object]] = []
        self.terminal_errors: list[str] = []

    def record_submission(
        self, *, prompt_ordinal: int, prompt_id: str, prompt_text: str, job_id: str
    ) -> None:
        self.job_records.append(
            {
                "prompt_ordinal": prompt_ordinal,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "job_id": job_id,
                "injection_monotonic_s": time.monotonic(),
                "injection_wall_time_s": time.time(),
                "response_id": None,
                "terminal_status": "submitted",
                "terminal_monotonic_s": None,
                "terminal_wall_time_s": None,
            }
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.audio_frames += 1
            self.audio_bytes += len(frame.audio)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.responses_done += 1
            pending = [
                record for record in self.job_records if record["terminal_status"] == "submitted"
            ]
            response_id = getattr(self.service, "step4_terminal_response_id", None)
            terminal_status = getattr(self.service, "step4_terminal_status", None)
            if not pending:
                self.terminal_errors.append("terminal response had no submitted prompt job")
            elif self.defer_terminal_identity:
                record = pending[0]
                record["terminal_status"] = "awaiting_trace"
                record["terminal_monotonic_s"] = time.monotonic()
                record["terminal_wall_time_s"] = time.time()
            elif not isinstance(response_id, str) or not response_id:
                self.terminal_errors.append("terminal response lacked a response_id")
            else:
                record = pending[0]
                record["response_id"] = response_id
                record["terminal_status"] = terminal_status or "unknown"
                record["terminal_monotonic_s"] = time.monotonic()
                record["terminal_wall_time_s"] = time.time()
        await self.push_frame(frame, direction)


def jobs_pass(records: list[dict[str, object]], prompts_sent: int, errors: list[str]) -> bool:
    if errors or prompts_sent <= 0 or len(records) != prompts_sent:
        return False
    ordinals = [record.get("prompt_ordinal") for record in records]
    response_ids = [record.get("response_id") for record in records]
    return bool(
        ordinals == list(range(1, prompts_sent + 1))
        and all(record.get("terminal_status") == "completed" for record in records)
        and all(isinstance(response_id, str) and response_id for response_id in response_ids)
        and len(set(response_ids)) == prompts_sent
    )


def read_health(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - fixed local URL
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("active_client") is not False:
        raise RuntimeError("single-client lock is occupied or health response is invalid")
    return payload


def backfill_terminal_identity_from_trace(
    records: list[dict[str, object]], trace_path: Path
) -> None:
    terminals = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("type") == "response.done"
    ]
    if len(terminals) != len(records):
        raise RuntimeError("Pipecat trace terminal count differs from prompt jobs")
    for record, terminal in zip(records, terminals, strict=True):
        response_id = terminal.get("response_id")
        status = terminal.get("status")
        if not isinstance(response_id, str) or not response_id:
            raise RuntimeError("Pipecat trace terminal lacks a response ID")
        record["response_id"] = response_id
        record["terminal_status"] = status if isinstance(status, str) else "completed"


async def run(args: argparse.Namespace) -> dict[str, object]:
    qualification_fixture = getattr(args, "qualification_fixture", None)
    step4c = qualification_fixture == QUALIFICATION_FIXTURE
    prompts = (L1_TEXT,) if step4c else PROMPTS
    warmup_seconds = float(getattr(args, "warmup_seconds", 0.0))
    trace_path_text = os.environ.get("NEMOTRON_VOICECHAT_PLAYOUT_TRACE", "").strip()
    if step4c and not trace_path_text:
        raise RuntimeError("Step 4c requires a configured native Pipecat playout trace path")
    health = read_health(args.health_url)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "health-immediately-before.json").write_text(
        json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    service_type = NemotronVoicechatLLMService if trace_path_text else QualificationService
    service = service_type(
        base_url=args.url,
        system_instruction=(SYSTEM_INSTRUCTION if step4c else "Answer briefly. Do not use tools."),
        tools=[],
    )
    sink = CaptureSink()
    sink.service = service  # type: ignore[assignment]
    sink.defer_terminal_identity = bool(trace_path_text)
    worker = PipelineWorker(
        Pipeline([service, sink]),
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
        params=PipelineParams(
            audio_in_sample_rate=INPUT_RATE,
            audio_out_sample_rate=24_000,
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    runner_task = asyncio.create_task(runner.run(), name="step4-pipecat-runner")
    await worker.queue_frame(LLMContextFrame(context=VoicechatLLMContext(messages=[], tools=[])))
    await service.wait_for_live_input_ready()
    started = time.monotonic() + 0.25
    total_frames = round(args.duration_seconds / FRAME_SECONDS)
    prompt_interval = max(1, round(args.typed_interval_seconds / FRAME_SECONDS))
    warmup_frames = round(warmup_seconds / FRAME_SECONDS)
    prompts_sent = 0
    silence = bytes(round(INPUT_RATE * FRAME_SECONDS) * 2)
    for frame_index in range(total_frames):
        deadline = started + frame_index * FRAME_SECONDS
        delay = deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        prompt_due = (
            frame_index >= warmup_frames and (frame_index - warmup_frames) % prompt_interval == 0
        )
        if prompt_due and (not step4c or prompts_sent == 0):
            prompt_index = prompts_sent % len(prompts)
            prompt = prompts[prompt_index]
            job_id = f"step4-{prompts_sent:04d}"
            sink.record_submission(
                prompt_ordinal=prompts_sent + 1,
                prompt_id=f"prompt-{prompt_index + 1:02d}",
                prompt_text=prompt,
                job_id=job_id,
            )
            await worker.queue_frame(VoicechatTypedInputFrame(text=prompt, job_id=job_id))
            prompts_sent += 1
        await worker.queue_frame(
            InputAudioRawFrame(audio=silence, sample_rate=INPUT_RATE, num_channels=1)
        )
    await asyncio.sleep(args.drain_seconds)
    await worker.stop_when_done()
    await runner_task
    if trace_path_text:
        if "{connection}" in trace_path_text:
            raise RuntimeError("native Step 4 Pipecat runner requires a concrete trace path")
        backfill_terminal_identity_from_trace(sink.job_records, Path(trace_path_text))
    trace_status = service.live_input_readiness_status().get("playout_trace")
    return {
        "schema": "nemotron_voicechat.step4_pipecat_soak.v3",
        "passed": bool(
            service._session_id
            and sink.responses_done == prompts_sent
            and jobs_pass(sink.job_records, prompts_sent, sink.terminal_errors)
            and (trace_status is None or trace_status.get("valid") is True)
        ),
        "session_id": service._session_id,
        "duration_seconds": args.duration_seconds,
        "warmup_seconds": warmup_seconds,
        "typed_interval_seconds": args.typed_interval_seconds,
        "drain_seconds": args.drain_seconds,
        "source_frames": total_frames,
        "qualification_fixture": qualification_fixture,
        "system_instruction": SYSTEM_INSTRUCTION if step4c else "Answer briefly. Do not use tools.",
        "prompts": list(prompts),
        "prompt_sha256": L1_SHA256 if step4c else None,
        "prompts_sent": prompts_sent,
        "responses_done": sink.responses_done,
        "job_records": sink.job_records,
        "terminal_errors": sink.terminal_errors,
        "audio_frames": sink.audio_frames,
        "audio_bytes": sink.audio_bytes,
        "playout_trace_enabled": bool(os.environ.get("NEMOTRON_VOICECHAT_PLAYOUT_TRACE", "")),
        "playout_trace_status": trace_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--health-url", default="http://127.0.0.1:8786/health")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--typed-interval-seconds", type=float, default=45.0)
    parser.add_argument("--drain-seconds", type=float, default=15.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--qualification-fixture", choices=(QUALIFICATION_FIXTURE,))
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.typed_interval_seconds <= 0 or args.drain_seconds < 0:
        parser.error("durations and intervals are invalid")
    if args.warmup_seconds < 0 or args.warmup_seconds >= args.duration_seconds:
        parser.error("warm-up must be non-negative and shorter than the duration")
    if args.qualification_fixture == QUALIFICATION_FIXTURE and (
        args.duration_seconds != STEP4C_DURATION_SECONDS
        or args.warmup_seconds != STEP4C_WARMUP_SECONDS
        or args.drain_seconds != STEP4C_DRAIN_SECONDS
    ):
        parser.error("Step 4c fixes warm-up/duration/drain at 10/120/15 seconds")
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        result = {
            "schema": "nemotron_voicechat.step4_pipecat_soak.v3",
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
