#!/usr/bin/env python3
"""CPU-only Pocket TTS worker for server-owned typed input."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .pocket_ipc import MAX_AUDIO_BYTES, read_json, write_frame, write_json

INPUT_SAMPLE_RATE = 16_000
POCKET_REPOSITORY_CACHE = "models--kyutai--pocket-tts-without-voice-cloning"
MODEL_REVISION = "d29db7978e464fb90cb3359ee0c69a273b9142cc"
VOICE_REVISION = "e041936c75475d350b405bc870bcf7c22da4e9e6"
EXPECTED_ASSETS = {
    f"snapshots/{MODEL_REVISION}/languages/english_2026-04/model.safetensors": (
        219_029_196,
        "be9c6b4876d3f30740a8225dfcaa2e43dc4aeb753c15272735bee16bbb4abb0a",
    ),
    f"snapshots/{MODEL_REVISION}/languages/english_2026-04/tokenizer.model": (
        59_339,
        "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6",
    ),
    f"snapshots/{VOICE_REVISION}/languages/english_2026-04/embeddings/alba.safetensors": (
        6_194_424,
        "69c32db63ca56843d994f81f343f62e0bf2d73f7e4c9bc73e44bb1110b1d8845",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_assets() -> dict[str, Any]:
    """Require the three qualified Pocket files at immutable snapshot paths."""

    if cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hub = Path(cache)
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        hub = hf_home / "hub"
    root = hub / POCKET_REPOSITORY_CACHE
    verified = {}
    for relative, (expected_bytes, expected_sha256) in EXPECTED_ASSETS.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"missing or wrong-sized qualified Pocket asset: {path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise RuntimeError(f"qualified Pocket asset hash mismatch: {path}")
        verified[relative] = {"bytes": expected_bytes, "sha256": actual}
    return verified


class SynthesisCancelled(Exception):
    """Raised when a client cancels synthesis between generated chunks."""


@dataclass(frozen=True)
class SynthesisResult:
    pcm16: bytes
    generated_seconds: float
    wall_seconds: float
    time_to_first_audio_seconds: float


class PocketBackend:
    """One prewarmed Pocket model shared by serialized worker requests."""

    def __init__(self, *, language: str, voice: str, quantize: bool, threads: int):
        self.language = language
        self.voice = voice
        self.quantize = quantize
        self.threads = threads
        self.model: Any | None = None
        self.voice_state: Any | None = None
        self.package_version: str | None = None
        self.assets: dict[str, Any] = {}

    def load_and_prewarm(self) -> None:
        import torch
        from pocket_tts import TTSModel

        self.assets = validate_assets()
        torch.set_num_threads(self.threads)
        torch.set_num_interop_threads(1)
        self.package_version = version("pocket-tts")
        model = TTSModel.load_model(language=self.language, quantize=self.quantize)
        voice_state = model.get_state_for_audio_prompt(self.voice)
        # Do not publish readiness until the entire generation path works.
        self.model = model
        self.voice_state = voice_state
        result = self.synthesize("Ready.", threading.Event())
        if not result.pcm16:
            self.model = None
            self.voice_state = None
            raise RuntimeError("Pocket TTS prewarm returned no audio")

    def synthesize(self, text: str, cancel_event: threading.Event) -> SynthesisResult:
        import numpy as np
        from scipy.signal import resample_poly

        model = self.model
        voice_state = self.voice_state
        if model is None or voice_state is None:
            raise RuntimeError("Pocket TTS worker is not ready")
        started = time.monotonic()
        first_audio_at: float | None = None
        chunks: list[Any] = []
        if cancel_event.is_set():
            raise SynthesisCancelled
        for chunk in model.generate_audio_stream(voice_state, text, copy_state=True):
            if cancel_event.is_set():
                raise SynthesisCancelled
            if first_audio_at is None:
                first_audio_at = time.monotonic()
            values = chunk.detach().to("cpu").float().numpy()
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if values.size and not np.isfinite(values).all():
                raise ValueError("Pocket TTS returned non-finite audio")
            if values.size:
                chunks.append(values)
        if cancel_event.is_set():
            raise SynthesisCancelled
        if not chunks:
            raise RuntimeError("Pocket TTS returned no audio")
        audio = np.concatenate(chunks)
        source_rate = int(model.sample_rate)
        if source_rate != INPUT_SAMPLE_RATE:
            divisor = math.gcd(source_rate, INPUT_SAMPLE_RATE)
            audio = resample_poly(
                audio,
                INPUT_SAMPLE_RATE // divisor,
                source_rate // divisor,
            ).astype(np.float32, copy=False)
        pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        if len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("Pocket TTS output exceeds the IPC audio limit")
        finished = time.monotonic()
        return SynthesisResult(
            pcm16=pcm,
            generated_seconds=len(pcm) / (2 * INPUT_SAMPLE_RATE),
            wall_seconds=finished - started,
            time_to_first_audio_seconds=(first_audio_at or finished) - started,
        )


class WorkerServer:
    def __init__(self, backend: PocketBackend):
        self.backend = backend
        self.cancel_events: dict[str, threading.Event] = {}
        self.generation_lock = asyncio.Lock()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_json(reader)
            operation = request.get("op")
            if operation == "ping":
                await write_json(
                    writer,
                    {
                        "status": "ready",
                        "pid": os.getpid(),
                        "package_version": self.backend.package_version,
                        "sample_rate": INPUT_SAMPLE_RATE,
                        "cpus": sorted(os.sched_getaffinity(0)),
                        "threads": self.backend.threads,
                        "assets": self.backend.assets,
                    },
                )
            elif operation == "cancel":
                job_id = request.get("job_id")
                event = self.cancel_events.get(job_id)
                if event is not None:
                    event.set()
                await write_json(writer, {"status": "ok", "cancelled": event is not None})
            elif operation == "synthesize":
                await self._synthesize(request, writer)
            else:
                await write_json(writer, {"status": "error", "error": "unknown_operation"})
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as exc:  # noqa: BLE001 - isolate errors at the IPC boundary
            try:
                await write_json(
                    writer,
                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                )
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _synthesize(
        self, request: dict[str, Any], writer: asyncio.StreamWriter
    ) -> None:
        job_id = request.get("job_id")
        text = request.get("text")
        if not isinstance(job_id, str) or not job_id or not isinstance(text, str) or not text:
            raise ValueError("synthesize requires non-empty job_id and text")
        if job_id in self.cancel_events:
            raise ValueError("duplicate active job_id")
        cancel_event = threading.Event()
        self.cancel_events[job_id] = cancel_event
        try:
            async with self.generation_lock:
                result = await asyncio.to_thread(self.backend.synthesize, text, cancel_event)
            await write_json(
                writer,
                {
                    "status": "ok",
                    "job_id": job_id,
                    "sample_rate": INPUT_SAMPLE_RATE,
                    "audio_bytes": len(result.pcm16),
                    "generated_seconds": result.generated_seconds,
                    "wall_seconds": result.wall_seconds,
                    "time_to_first_audio_seconds": result.time_to_first_audio_seconds,
                },
            )
            await write_frame(writer, result.pcm16)
        except SynthesisCancelled:
            await write_json(writer, {"status": "cancelled", "job_id": job_id})
        finally:
            self.cancel_events.pop(job_id, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--language", default="english_2026-04")
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpus", default="7,8,9,15")
    parser.add_argument("--no-quantize", action="store_true")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    cpus = {int(item) for item in args.cpus.split(",")}
    if not cpus:
        raise ValueError("Pocket TTS CPU affinity may not be empty")
    os.sched_setaffinity(0, cpus)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["GOMP_CPU_AFFINITY"] = " ".join(str(cpu) for cpu in sorted(cpus))
    backend = PocketBackend(
        language=args.language,
        voice=args.voice,
        quantize=not args.no_quantize,
        threads=args.threads,
    )
    await asyncio.to_thread(backend.load_and_prewarm)
    path = Path(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(WorkerServer(backend).handle, path=path)
    os.chmod(path, 0o600)
    try:
        async with server:
            await server.serve_forever()
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
