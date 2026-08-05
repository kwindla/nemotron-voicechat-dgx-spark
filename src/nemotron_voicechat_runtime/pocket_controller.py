"""Lifecycle and client for the isolated Pocket TTS process."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pocket_ipc import MAX_AUDIO_BYTES, read_frame, read_json, write_json


class PocketWorkerError(RuntimeError):
    """The isolated worker was unavailable or rejected a request."""


class PocketSynthesisCancelled(PocketWorkerError):
    """The worker acknowledged cancellation of a synthesis request."""


@dataclass(frozen=True)
class PocketSynthesis:
    pcm16: bytes
    sample_rate: int
    generated_seconds: float
    wall_seconds: float
    time_to_first_audio_seconds: float


class PocketWorkerManager:
    """Supervise one prewarmed CPU-only Pocket TTS worker."""

    def __init__(
        self,
        *,
        executable: str = "/opt/pocket-tts/bin/python",
        socket_path: Path | None = None,
        language: str = "english_2026-04",
        voice: str = "alba",
        threads: int = 4,
        cpus: tuple[int, ...] = (7, 8, 9, 15),
        startup_timeout: float = 120.0,
    ):
        self.executable = executable
        self.socket_path = socket_path or Path(tempfile.gettempdir()) / "voicechat-pocket.sock"
        self.language = language
        self.voice = voice
        self.threads = threads
        self.cpus = cpus
        codec_cpus = {
            int(value) for value in os.environ.get("EA_CPU_CODEC_CORES", "5,6").split(",")
        }
        if not cpus or codec_cpus & set(cpus):
            raise ValueError(
                "Pocket TTS CPU affinity must be non-empty and disjoint from codec cores"
            )
        self.startup_timeout = startup_timeout
        self.process: asyncio.subprocess.Process | None = None
        self._ready: dict[str, Any] | None = None
        self._operation_lock = asyncio.Lock()
        self._restart_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        process = self.process
        return process is not None and process.returncode is None and self._ready is not None

    @property
    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "pid": self.process.pid if self.ready and self.process else None,
            "socket": str(self.socket_path),
            "language": self.language,
            "voice": self.voice,
            **(self._ready or {}),
        }

    def _worker_environment(self) -> dict[str, str]:
        """Build a narrow environment without inherited CUDA/Python library paths."""

        allowed = {
            "HOME",
            "LANG",
            "LC_ALL",
            "TZ",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "XDG_CACHE_HOME",
        }
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.update(
            {
                "PATH": str(Path(self.executable).parent) + ":/usr/bin:/bin",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        return environment

    async def start(self) -> None:
        if self.ready:
            return
        async with self._restart_lock:
            if self.ready:
                return
            await self.stop()
            self.socket_path.unlink(missing_ok=True)
            self.process = await asyncio.create_subprocess_exec(
                self.executable,
                "-m",
                "nemotron_voicechat_runtime.pocket_worker",
                "--socket",
                str(self.socket_path),
                "--language",
                self.language,
                "--voice",
                self.voice,
                "--threads",
                str(self.threads),
                "--cpus",
                ",".join(str(cpu) for cpu in self.cpus),
                env=self._worker_environment(),
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            deadline = time.monotonic() + self.startup_timeout
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    raise PocketWorkerError(
                        f"Pocket worker exited during startup: {self.process.returncode}"
                    )
                try:
                    status, _ = await self._request({"op": "ping"}, expect_audio=False)
                    if status.get("status") == "ready":
                        self._ready = status
                        return
                except (ConnectionError, FileNotFoundError, OSError) as exc:
                    last_error = exc
                await asyncio.sleep(0.1)
            await self.stop()
            raise PocketWorkerError(f"Pocket worker startup timed out: {last_error}")

    async def stop(self) -> None:
        process, self.process = self.process, None
        self._ready = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        self.socket_path.unlink(missing_ok=True)

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    async def synthesize(
        self,
        text: str,
        *,
        job_id: str,
        cancel_event: asyncio.Event,
    ) -> PocketSynthesis:
        if not self.ready:
            await self.start()
        async with self._operation_lock:
            request = asyncio.create_task(
                self._request(
                    {"op": "synthesize", "job_id": job_id, "text": text},
                    expect_audio=True,
                )
            )
            cancellation = asyncio.create_task(cancel_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {request, cancellation}, return_when=asyncio.FIRST_COMPLETED
                )
                if cancellation in done and cancel_event.is_set() and not request.done():
                    await self.cancel(job_id)
                try:
                    status, audio = await request
                except (ConnectionError, asyncio.IncompleteReadError, OSError) as exc:
                    self._ready = None
                    raise PocketWorkerError(f"Pocket worker connection failed: {exc}") from exc
            finally:
                cancellation.cancel()
                await asyncio.gather(cancellation, return_exceptions=True)
            if status.get("status") == "cancelled":
                raise PocketSynthesisCancelled(job_id)
            if status.get("status") != "ok" or audio is None:
                if self.process and self.process.returncode is not None:
                    self._ready = None
                raise PocketWorkerError(str(status.get("error") or "Pocket synthesis failed"))
            return PocketSynthesis(
                pcm16=audio,
                sample_rate=int(status["sample_rate"]),
                generated_seconds=float(status["generated_seconds"]),
                wall_seconds=float(status["wall_seconds"]),
                time_to_first_audio_seconds=float(status["time_to_first_audio_seconds"]),
            )

    async def cancel(self, job_id: str) -> None:
        try:
            await self._request({"op": "cancel", "job_id": job_id}, expect_audio=False)
        except (ConnectionError, OSError):
            self._ready = None

    async def _request(
        self, request: dict[str, Any], *, expect_audio: bool
    ) -> tuple[dict[str, Any], bytes | None]:
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        try:
            await write_json(writer, request)
            status = await read_json(reader)
            audio = None
            if expect_audio and status.get("status") == "ok":
                expected = int(status.get("audio_bytes", -1))
                audio = await read_frame(reader, limit=MAX_AUDIO_BYTES)
                if len(audio) != expected:
                    raise PocketWorkerError("Pocket worker audio length mismatch")
            return status, audio
        finally:
            writer.close()
            await writer.wait_closed()
