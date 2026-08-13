from __future__ import annotations

import base64
import copy
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
from pipecat.frames.frames import (
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)

from nemotron_voicechat_pipecat import events
from nemotron_voicechat_pipecat.llm import (
    PLAYOUT_TRACE_ENV,
    PLAYOUT_TRACE_SCHEMA,
    NemotronVoicechatLLMService,
    _PlayoutTraceWriter,
    _TracedNemotronVoicechatLLMService,
)


def _load_analyzer() -> ModuleType:
    path = Path("tools/qualification/stratified_latency_analyzer.py")
    spec = importlib.util.spec_from_file_location("step2_stratified_latency_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RecordingService(NemotronVoicechatLLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames = []

    async def push_frame(self, frame, direction=None):
        self.frames.append(frame)

    async def stop_ttfb_metrics(self):
        pass

    async def stop_all_metrics(self):
        pass


class _RecordingTracedService(_TracedNemotronVoicechatLLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames = []

    async def push_frame(self, frame, direction=None):
        self.frames.append(frame)

    async def stop_ttfb_metrics(self):
        pass

    async def stop_all_metrics(self):
        pass


class _BlockingStream:
    def __init__(self, *, fail: bool = False):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self.closed = False
        self.payloads: list[str] = []

    def write(self, payload: str) -> int:
        self.entered.set()
        self.release.wait(5)
        if self.fail:
            raise OSError("synthetic disk failure")
        self.payloads.append(payload)
        return len(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _CloseFailingStream:
    def __init__(self):
        self.payloads: list[str] = []
        self.flushed = False

    def write(self, payload: str) -> int:
        self.payloads.append(payload)
        return len(payload)

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        raise OSError("synthetic close-after-flush failure")


class _FailingFileStream:
    def __init__(self, stream, failure: str):
        self._stream = stream
        self._failure = failure

    def write(self, payload: str) -> int:
        if self._failure == "write":
            raise OSError("synthetic staging write failure")
        return self._stream.write(payload)

    def flush(self) -> None:
        self._stream.flush()
        if self._failure == "flush":
            raise OSError("synthetic staging flush failure")

    def fileno(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:
        self._stream.close()
        if self._failure == "close":
            raise OSError("synthetic staging close failure")


def _audio_event(response_id: str, pcm: bytes) -> dict[str, object]:
    return {
        "type": "response.output_audio.delta",
        "response_id": response_id,
        "encoding": "pcm16",
        "sample_rate": events.OUTPUT_SAMPLE_RATE,
        "channels": 1,
        "delta": base64.b64encode(pcm).decode(),
    }


def _stamp(event: dict[str, object], timestamp: float) -> dict[str, object]:
    return {**event, "_voicechat_received_monotonic_s": timestamp}


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_playout_trace_writer_batches_and_writes_terminal_status(tmp_path) -> None:
    times = iter((10.25, 10.5, 10.75))
    path = tmp_path / "playout.jsonl"
    writer = _PlayoutTraceWriter(path, clock=lambda: next(times), batch_records=8)
    writer.record(
        "response.output_audio.delta",
        timestamp_field="client_received_monotonic_s",
        response_id="response_1",
        ordinal=1,
        sample_count=1764,
    )
    writer.record(
        "voicechat.playout.release",
        timestamp_field="client_release_monotonic_s",
        response_id="response_1",
        reason="threshold",
        frames_released=1,
    )
    status = writer.close()

    records = _records(path)
    assert records[0]["client_received_monotonic_s"] == 10.25
    assert records[1]["client_release_monotonic_s"] == 10.5
    assert records[-1]["type"] == "voicechat.playout.trace_status"
    assert records[-1]["valid"] is True
    assert all(record["trace_schema"] == PLAYOUT_TRACE_SCHEMA for record in records)
    assert status["valid"] is True
    assert status["writer_alive"] is False


def test_writer_queue_overflow_invalidates_without_blocking_media(tmp_path) -> None:
    stream = _BlockingStream()
    writer = _PlayoutTraceWriter(
        tmp_path / "overflow.jsonl", stream=stream, queue_records=1, batch_records=1
    )
    assert writer.record("one")
    assert stream.entered.wait(1)
    assert writer.record("two")
    started = time.perf_counter()
    assert writer.record("overflow") is False
    assert time.perf_counter() - started < 0.1
    assert writer.status()["invalid_reason"] == "queue-overflow"
    stream.release.set()
    status = writer.close(timeout=1)
    assert status["valid"] is False
    assert status["dropped_records"] == 1


def test_writer_disk_error_is_status_only(tmp_path) -> None:
    stream = _BlockingStream(fail=True)
    stream.release.set()
    writer = _PlayoutTraceWriter(tmp_path / "disk.jsonl", stream=stream)
    assert writer.record("record")
    assert stream.entered.wait(1)
    deadline = time.monotonic() + 1
    while writer.status()["error"] is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert writer.record("after-error") is False
    status = writer.close(timeout=0.5)
    assert status["valid"] is False
    assert status["invalid_reason"] == "writer-error"
    assert "synthetic disk failure" in status["error"]


def test_writer_close_is_bounded_when_storage_stalls(tmp_path) -> None:
    stream = _BlockingStream()
    writer = _PlayoutTraceWriter(
        tmp_path / "stalled.jsonl", stream=stream, queue_records=2, batch_records=1
    )
    assert writer.record("record")
    assert stream.entered.wait(1)
    started = time.perf_counter()
    status = writer.close(timeout=0.05)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.25
    assert status["valid"] is False
    assert status["invalid_reason"] == "close-timeout"
    stream.release.set()


def test_close_after_flush_failure_never_publishes_acceptable_artifact(tmp_path) -> None:
    path = tmp_path / "close-failed.jsonl"
    stream = _CloseFailingStream()
    writer = _PlayoutTraceWriter(path, stream=stream)
    assert writer.record("voicechat.playout.config")

    status = writer.close(timeout=1)

    assert stream.flushed is True
    assert status["valid"] is False
    assert status["invalid_reason"] == "close-error"
    assert status["artifact_published"] is False
    assert not path.exists()
    persisted = [json.loads(line) for payload in stream.payloads for line in payload.splitlines()]
    assert persisted[-1]["type"] == "voicechat.playout.trace_status"


def test_atomic_no_replace_allows_only_one_racing_writer(tmp_path) -> None:
    path = tmp_path / "racing.jsonl"
    writers = [_PlayoutTraceWriter(path), _PlayoutTraceWriter(path)]
    for marker, writer in enumerate(writers, 1):
        assert writer.record("race.marker", marker=marker)
    barrier = threading.Barrier(3)
    statuses: list[dict] = []

    def close_writer(writer: _PlayoutTraceWriter) -> None:
        barrier.wait()
        statuses.append(writer.close(timeout=2))

    threads = [threading.Thread(target=close_writer, args=(writer,)) for writer in writers]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(3)

    assert len(statuses) == 2
    assert sum(status["artifact_published"] for status in statuses) == 1
    assert sum(status["valid"] for status in statuses) == 1
    assert {record["marker"] for record in _records(path) if "marker" in record} in (
        {1},
        {2},
    )
    assert not list(tmp_path.glob(".racing.jsonl.*.pending"))


def test_late_destination_and_publish_failure_preserve_destination_and_clean_staging(
    monkeypatch, tmp_path
) -> None:
    late_path = tmp_path / "late.jsonl"
    late_writer = _PlayoutTraceWriter(late_path)
    assert late_writer.record("marker", value="writer")
    late_path.write_text("sentinel\n", encoding="utf-8")

    late_status = late_writer.close(timeout=1)

    assert late_status["valid"] is False
    assert late_status["invalid_reason"] == "publish-error"
    assert late_path.read_text(encoding="utf-8") == "sentinel\n"
    assert not list(tmp_path.glob(".late.jsonl.*.pending"))

    failed_path = tmp_path / "publish-failed.jsonl"
    failed_writer = _PlayoutTraceWriter(failed_path)
    assert failed_writer.record("marker", value="writer")

    def fail_link(_source, _destination) -> None:
        raise OSError("synthetic link failure")

    monkeypatch.setattr(os, "link", fail_link)
    failed_status = failed_writer.close(timeout=1)
    assert failed_status["invalid_reason"] == "publish-error"
    assert not failed_path.exists()
    assert not list(tmp_path.glob(".publish-failed.jsonl.*.pending"))


@pytest.mark.parametrize("failure", ["write", "flush", "close"])
def test_handled_staging_io_failure_unlinks_pending_file(tmp_path, failure: str) -> None:
    path = tmp_path / f"{failure}-failed.jsonl"
    writer = _PlayoutTraceWriter(path)
    writer._stream = _FailingFileStream(writer._stream, failure)
    assert writer.record("marker", value="writer")

    status = writer.close(timeout=1)

    expected_reason = "close-error" if failure == "close" else "writer-error"
    assert status["invalid_reason"] == expected_reason
    assert not path.exists()
    assert not list(tmp_path.glob(f".{failure}-failed.jsonl.*.pending"))


def test_publication_durability_failure_rolls_back_link_and_cleans_staging(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "durability-failed.jsonl"
    writer = _PlayoutTraceWriter(path)
    assert writer.record("marker", value="writer")
    real_fsync_directory = writer._fsync_directory
    calls = 0

    def fail_first_directory_sync(directory: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic directory fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(writer, "_fsync_directory", fail_first_directory_sync)
    status = writer.close(timeout=1)

    assert status["invalid_reason"] == "publish-error"
    assert not path.exists()
    assert not list(tmp_path.glob(".durability-failed.jsonl.*.pending"))


def test_killed_writer_pending_file_is_quarantined_and_never_published(tmp_path) -> None:
    path = tmp_path / "killed.jsonl"
    code = (
        "import sys,time\n"
        "from nemotron_voicechat_pipecat.llm import _PlayoutTraceWriter\n"
        "writer=_PlayoutTraceWriter(sys.argv[1])\n"
        "stream=writer._stream\n"
        "class BlockedWrite:\n"
        " def write(self,payload):\n"
        "  print(writer._staging_path,flush=True)\n"
        "  time.sleep(30)\n"
        "  return stream.write(payload)\n"
        " def flush(self): return stream.flush()\n"
        " def close(self): return stream.close()\n"
        " def fileno(self): return stream.fileno()\n"
        "writer._stream=BlockedWrite()\n"
        "writer.record('killed.marker', marker='orphan')\n"
        "time.sleep(30)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pending = Path(process.stdout.readline().strip())
    assert pending.exists()
    process.kill()
    process.wait(timeout=5)
    assert not path.exists()

    recovery = _PlayoutTraceWriter(path)
    assert recovery.record("recovery.marker", marker="recovered")
    status = recovery.close(timeout=1)

    quarantine = tmp_path / _PlayoutTraceWriter._QUARANTINE_DIRECTORY
    assert status["artifact_published"] is True
    assert [record.get("marker") for record in _records(path) if "marker" in record] == [
        "recovered"
    ]
    assert not list(tmp_path.glob(".killed.jsonl.*.pending"))
    assert any(item.name == f"{pending.name}.orphan" for item in quarantine.iterdir())


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_traced_service_rejects_invalid_close_timeout_at_construction(
    monkeypatch, tmp_path, timeout: str
) -> None:
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(tmp_path / f"invalid-{timeout}.jsonl"))
    monkeypatch.setenv("NEMOTRON_VOICECHAT_PLAYOUT_TRACE_CLOSE_TIMEOUT", timeout)
    with pytest.raises(ValueError, match="finite and positive"):
        NemotronVoicechatLLMService()


@pytest.mark.asyncio
async def test_trace_off_selects_original_branch_free_hot_methods(monkeypatch) -> None:
    monkeypatch.delenv(PLAYOUT_TRACE_ENV, raising=False)
    service = _RecordingService(prebuffer_ms=0)
    pcm = b"\x02\x00" * 80
    await service._response_created({"response_id": "response_1"})
    await service._audio_delta(_audio_event("response_1", pcm))

    assert type(service) is _RecordingService
    assert service._audio_delta.__func__ is NemotronVoicechatLLMService._audio_delta
    assert service._release_playout_buffer.__func__ is (
        NemotronVoicechatLLMService._release_playout_buffer
    )
    assert service._clear_playout_buffer.__func__ is (
        NemotronVoicechatLLMService._clear_playout_buffer
    )
    assert service._receive_loop.__func__ is NemotronVoicechatLLMService._receive_loop
    assert not hasattr(service, "_playout_trace")
    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
    ]
    assert service.frames[-1].audio == pcm


def test_public_constructor_selects_traced_variant_once(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(tmp_path / "selected.jsonl"))
    service = NemotronVoicechatLLMService()
    assert type(service) is _TracedNemotronVoicechatLLMService
    assert service._audio_delta.__func__ is _TracedNemotronVoicechatLLMService._audio_delta
    service._playout_trace.close()


def test_shallow_copy_is_rejected_with_trace_off_and_after_env_change(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(PLAYOUT_TRACE_ENV, raising=False)
    service = NemotronVoicechatLLMService()
    with pytest.raises(TypeError, match="cannot be shallow-copied"):
        copy.copy(service)

    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(tmp_path / "late-trace.jsonl"))
    with pytest.raises(TypeError, match="cannot be shallow-copied"):
        copy.copy(service)
    assert type(service) is NemotronVoicechatLLMService
    assert not hasattr(service, "_playout_trace")


def test_shallow_copy_of_traced_service_is_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(tmp_path / "traced-copy.jsonl"))
    service = NemotronVoicechatLLMService()
    try:
        with pytest.raises(TypeError, match="cannot be shallow-copied"):
            copy.copy(service)
    finally:
        service._playout_trace.close()


def test_deepcopy_and_pickle_are_explicitly_rejected_for_every_service_variant(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(PLAYOUT_TRACE_ENV, raising=False)
    ordinary = NemotronVoicechatLLMService()
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(ordinary)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(ordinary)

    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(tmp_path / "late-deepcopy.jsonl"))
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(ordinary)

    traced = NemotronVoicechatLLMService()
    try:
        with pytest.raises(TypeError, match="cannot be deep-copied"):
            copy.deepcopy(traced)
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(traced)
    finally:
        traced._playout_trace.close()


def test_subclass_rejects_shallow_and_deep_copy(monkeypatch) -> None:
    monkeypatch.delenv(PLAYOUT_TRACE_ENV, raising=False)
    service = _RecordingService()
    with pytest.raises(TypeError, match="cannot be shallow-copied"):
        copy.copy(service)
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(service)


@pytest.mark.asyncio
async def test_traced_cleanup_closes_writer_off_event_loop(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(tmp_path / "cleanup.jsonl"))
    service = _RecordingTracedService()
    event_loop_thread = threading.get_ident()
    close_threads: list[int] = []
    original_close = service._playout_trace.close

    def observed_close(*, timeout: float):
        close_threads.append(threading.get_ident())
        return original_close(timeout=timeout)

    service._playout_trace.close = observed_close

    async def no_disconnect() -> None:
        pass

    service._disconnect = no_disconnect
    await service.cleanup()

    assert close_threads
    assert close_threads[0] != event_loop_thread
    assert service._playout_trace.status()["writer_alive"] is False


@pytest.mark.asyncio
async def test_receipt_clock_is_sampled_at_iterator_boundary_and_shared_with_liveness(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "receipt.jsonl"
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(path))
    service = _RecordingTracedService()

    class _WebSocket:
        def __aiter__(self):
            self.messages = iter(
                (
                    json.dumps({"type": "response.created", "response_id": "response_1"}),
                    json.dumps(
                        {
                            "type": "voicechat.metrics",
                            "response_id": "response_1",
                            "frame": 1,
                        }
                    ),
                )
            )
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise StopAsyncIteration from None

    service._websocket = _WebSocket()
    service._disconnecting = True
    with monkeypatch.context() as receipt_patch:
        times = iter((101.0, 102.0))
        receipt_patch.setattr(
            "nemotron_voicechat_pipecat.llm.time.monotonic", lambda: next(times)
        )
        await service._receive_loop()
    service._playout_trace.close()

    records = _records(path)
    received = [
        record["client_received_monotonic_s"]
        for record in records
        if "client_received_monotonic_s" in record
    ]
    assert received == [101.0, 102.0]
    assert service._last_server_event_at == 102.0


@pytest.mark.asyncio
async def test_lifecycle_filter_precedes_single_decode(monkeypatch, tmp_path) -> None:
    path = tmp_path / "lifecycle.jsonl"
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(path))
    service = _RecordingTracedService(prebuffer_ms=0)
    await service._handle_server_event(
        _stamp({"type": "response.created", "response_id": "response_1"}, 1.0)
    )
    original_decode = events.decode_audio_delta
    calls = 0

    def counted_decode(event):
        nonlocal calls
        calls += 1
        return original_decode(event)

    monkeypatch.setattr(events, "decode_audio_delta", counted_decode)
    malformed_stale = {
        "type": "response.output_audio.delta",
        "response_id": "stale",
        "encoding": "opus",
        "delta": "not-base64",
    }
    await service._handle_server_event(_stamp(malformed_stale, 1.1))
    service._response_cancelling = True
    await service._handle_server_event(
        _stamp({**malformed_stale, "response_id": "response_1"}, 1.2)
    )
    service._response_cancelling = False
    pcm = b"\x03\x00" * 80
    await service._handle_server_event(_stamp(_audio_event("response_1", pcm), 1.3))
    service._playout_trace.close()

    assert calls == 1
    deltas = [r for r in _records(path) if r["type"] == "response.output_audio.delta"]
    assert len(deltas) == 1
    assert deltas[0]["client_received_monotonic_s"] == 1.3
    assert service.frames[-1].audio == pcm


@pytest.mark.asyncio
async def test_traced_text_identity_records_only_current_accepted_response(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "response-text.jsonl"
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(path))
    service = _RecordingTracedService()
    await service._handle_server_event(
        _stamp({"type": "response.created", "response_id": "response_1"}, 1.0)
    )
    await service._handle_server_event(
        _stamp(
            {
                "type": "response.output_text.delta",
                "response_id": "response_1",
                "delta": "owned response text",
            },
            1.1,
        )
    )
    await service._handle_server_event(
        _stamp(
            {
                "type": "response.output_text.delta",
                "response_id": "stale",
                "delta": "stale text",
            },
            1.2,
        )
    )
    service._response_cancelling = True
    await service._handle_server_event(
        _stamp(
            {
                "type": "response.output_text.delta",
                "response_id": "response_1",
                "delta": "cancelled text",
            },
            1.3,
        )
    )
    service._playout_trace.close()

    text_records = [
        record for record in _records(path) if record["type"] == "response.output_text.delta"
    ]
    assert text_records == [
        {
            "trace_schema": PLAYOUT_TRACE_SCHEMA,
            "type": "response.output_text.delta",
            "client_received_monotonic_s": 1.1,
            "response_id": "response_1",
            "ordinal": 1,
            "delta": "owned response text",
        }
    ]


@pytest.mark.asyncio
async def test_typed_speech_without_active_response_does_not_fabricate_clear(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "idle-typed.jsonl"
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(path))
    service = _RecordingTracedService()
    await service._handle_server_event(
        _stamp(
            {"type": "input_audio_buffer.speech_started", "source": "typed"},
            time.monotonic(),
        )
    )
    service._playout_trace.close()

    records = _records(path)
    assert not any(record["type"] == "voicechat.playout.release" for record in records)
    validation = _load_analyzer().validate_observed_playout_trace(records)
    assert validation["valid"] is True
    assert validation["response_count"] == 0


@pytest.mark.asyncio
async def test_trace_records_release_push_done_and_interruption_linkage(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "playout.jsonl"
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(path))
    service = _RecordingTracedService(prebuffer_ms=160)
    pcm = b"\x01\x00" * int(events.OUTPUT_SAMPLE_RATE * 0.08)
    metrics = {
        "type": "voicechat.metrics",
        "event_id": "evt_test_metrics",
        "session_id": "session_test",
        "response_id": "response_1",
        "turn_id": "turn_1",
        "audio_delivered": True,
        "transport_frame": 1,
        "inference_ms": 80.0,
        "server_step_ms": 81.0,
        "budget_ms": 80.0,
        "over_budget": True,
        "input_audio": {"samples": 1280},
        "output_audio": {"samples": len(pcm) // 2},
        "model_output_audio": {"samples": len(pcm) // 2},
        "turn_state": {},
        "step9_capture_fallback": {},
    }
    server_events = [
        {"type": "response.created", "response_id": "response_1"},
        {**metrics, "frame": 1},
        _audio_event("response_1", pcm),
        {**metrics, "frame": 2},
        _audio_event("response_1", pcm),
        {"type": "response.done", "response_id": "response_1", "status": "completed"},
    ]
    for event in server_events:
        await service._handle_server_event(_stamp(event, time.monotonic()))
    service._playout_trace.close()

    records = _records(path)
    release = next(record for record in records if record["type"] == "voicechat.playout.release")
    assert release["reason"] == "threshold"
    assert release["ordinals"] == [1, 2]
    pushes = [r for r in records if r["type"] == "voicechat.playout.downstream_push"]
    assert [record["ordinal"] for record in pushes] == [1, 2]
    report = _load_analyzer().analyze_events(records)
    assert report["queue_diagnostics"]["delivered_frame_reconciliation"]["provable"]

    clear_path = tmp_path / "clear.jsonl"
    monkeypatch.setenv(PLAYOUT_TRACE_ENV, str(clear_path))
    clear_service = _RecordingTracedService(prebuffer_ms=160)
    await clear_service._handle_server_event(
        _stamp(
            {"type": "response.created", "response_id": "response_2"},
            time.monotonic(),
        )
    )
    await clear_service._handle_server_event(
        _stamp(_audio_event("response_2", pcm), time.monotonic())
    )
    await clear_service._handle_server_event(
        _stamp(
            {"type": "input_audio_buffer.speech_started", "source": "typed"},
            time.monotonic(),
        )
    )
    clear_service._playout_trace.close()
    cleared = next(
        record for record in _records(clear_path) if record.get("reason") == "interruption-clear"
    )
    assert cleared["ordinals"] == [1]
    assert cleared["frames_cleared"] == 1
