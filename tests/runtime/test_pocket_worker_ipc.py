"""Safe IPC and lifecycle tests for the isolated Pocket TTS worker."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from nemotron_voicechat_runtime import pocket_worker
from nemotron_voicechat_runtime.pocket_controller import (
    PocketSynthesisCancelled,
    PocketWorkerError,
    PocketWorkerManager,
)
from nemotron_voicechat_runtime.pocket_ipc import read_json, write_frame, write_json
from nemotron_voicechat_runtime.protocol import RealtimeProtocolSession, protocol_capabilities
from nemotron_voicechat_runtime.server import typed_input_trailing_silence_seconds


def ready_manager(socket_path):
    manager = PocketWorkerManager(socket_path=socket_path)
    manager.process = SimpleNamespace(returncode=None, pid=1234)
    manager._ready = {"status": "ready"}
    return manager


@pytest.mark.asyncio
async def test_manager_reads_length_prefixed_json_and_raw_pcm(tmp_path):
    socket = tmp_path / "pocket.sock"
    pcm = b"\x01\x00" * 320

    async def handler(reader, writer):
        request = await read_json(reader)
        assert request == {"op": "synthesize", "job_id": "job-1", "text": "Hello"}
        await write_json(
            writer,
            {
                "status": "ok",
                "job_id": "job-1",
                "sample_rate": 16_000,
                "audio_bytes": len(pcm),
                "generated_seconds": 0.02,
                "wall_seconds": 0.01,
                "time_to_first_audio_seconds": 0.005,
            },
        )
        await write_frame(writer, pcm)
        writer.close()

    server = await asyncio.start_unix_server(handler, path=socket)
    manager = ready_manager(socket)
    async with server:
        result = await manager.synthesize("Hello", job_id="job-1", cancel_event=asyncio.Event())
    assert result.pcm16 == pcm
    assert result.sample_rate == 16_000


@pytest.mark.asyncio
async def test_cancellation_uses_a_second_connection_and_returns_typed_disposition(tmp_path):
    socket = tmp_path / "pocket.sock"
    cancelled = asyncio.Event()

    async def handler(reader, writer):
        request = await read_json(reader)
        if request["op"] == "cancel":
            cancelled.set()
            await write_json(writer, {"status": "ok", "cancelled": True})
        else:
            await cancelled.wait()
            await write_json(writer, {"status": "cancelled", "job_id": "job-2"})
        writer.close()

    server = await asyncio.start_unix_server(handler, path=socket)
    manager = ready_manager(socket)
    cancel_event = asyncio.Event()
    async with server:
        task = asyncio.create_task(
            manager.synthesize("Slow", job_id="job-2", cancel_event=cancel_event)
        )
        await asyncio.sleep(0.01)
        cancel_event.set()
        with pytest.raises(PocketSynthesisCancelled):
            await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_worker_crash_degrades_only_typed_input_and_allows_restart(tmp_path, monkeypatch):
    socket = tmp_path / "pocket.sock"

    async def crash(_reader, writer):
        writer.close()

    server = await asyncio.start_unix_server(crash, path=socket)
    manager = ready_manager(socket)
    async with server:
        with pytest.raises(PocketWorkerError, match="Pocket worker connection failed"):
            await manager.synthesize("Hello", job_id="job-crash", cancel_event=asyncio.Event())
    assert manager.health["ready"] is False

    restarted = False

    async def fake_start():
        nonlocal restarted
        restarted = True

    monkeypatch.setattr(manager, "start", fake_start)
    with pytest.raises(PocketWorkerError, match="Pocket worker connection failed"):
        await manager.synthesize("Retry", job_id="job-retry", cancel_event=asyncio.Event())
    assert restarted


def test_worker_environment_scrubs_cuda_and_python_library_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cuda")
    monkeypatch.setenv("CUDA_HOME", "/cuda")
    monkeypatch.setenv("PYTHONPATH", "/gpu/site-packages")
    monkeypatch.setenv("HF_HOME", "/models")
    environment = PocketWorkerManager(socket_path=tmp_path / "pocket.sock")._worker_environment()
    assert "LD_LIBRARY_PATH" not in environment
    assert "CUDA_HOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["HF_HOME"] == "/models"
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_worker_affinity_must_not_overlap_codec_cores(monkeypatch, tmp_path):
    monkeypatch.setenv("EA_CPU_CODEC_CORES", "5,6")
    with pytest.raises(ValueError, match="disjoint from codec cores"):
        PocketWorkerManager(socket_path=tmp_path / "pocket.sock", cpus=(6, 7))


def test_worker_assets_are_verified_at_immutable_snapshot_paths(monkeypatch, tmp_path):
    payload = b"qualified"
    relative = "snapshots/revision/model.safetensors"
    root = tmp_path / "hub" / pocket_worker.POCKET_REPOSITORY_CACHE
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(
        pocket_worker,
        "EXPECTED_ASSETS",
        {relative: (len(payload), hashlib.sha256(payload).hexdigest())},
    )
    assert pocket_worker.validate_assets()[relative]["bytes"] == len(payload)
    path.write_bytes(b"corruptxx")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        pocket_worker.validate_assets()


def test_eou_closure_has_margin_and_model_frame_alignment():
    seconds = typed_input_trailing_silence_seconds(rnnt_eou_frames=20, margin=1.25)
    assert seconds == pytest.approx(2.08)
    assert seconds > 20 * 0.08
    assert seconds % 0.08 == pytest.approx(0.0)


def test_typed_rnnt_transcript_is_tagged_with_source_and_job_id():
    protocol = RealtimeProtocolSession(
        "typed",
        protocol_capabilities(
            function_call_timeout_seconds=30,
            max_session_model_frames=12_000,
            typed_input=True,
        ),
    )
    protocol.set_input_source("typed", "job-3")
    event = protocol._new_turn()
    assert event["source"] == "typed"
    assert event["job_id"] == "job-3"
    transcript = protocol._update_turn_transcript("Exact text")
    assert transcript[0]["source"] == "typed"
    assert transcript[0]["job_id"] == "job-3"
