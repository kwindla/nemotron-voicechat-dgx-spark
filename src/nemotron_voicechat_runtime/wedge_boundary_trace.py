"""Bounded nonblocking B0--B9 diagnostic records for the pairing wedge.

The trace is inert unless ``NEMOTRON_WEDGE_TRACE=1``.  When enabled, every
process owns one fixed-size mmap sidecar, so recording never waits on a lock,
pipe, socket, or another process.  The ring saturates to retain the earliest
causal prefix; later records are counted as dropped.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import threading
import time
import uuid
from asyncio import current_task
from pathlib import Path
from typing import Any

BOUNDARIES = tuple(f"B{index}" for index in range(10))
SCHEMA_VERSION = 2
RECORD_BYTES = 512
DEFAULT_CAPACITY = 4096
_TRUE = {"1", "true", "yes", "on"}


class _SidecarRing:
    def __init__(self, directory: Path, capacity: int) -> None:
        self.capacity = capacity
        self.sequence = 0
        self.dropped = 0
        directory.mkdir(parents=True, exist_ok=True)
        nonce = os.environ.get("NEMOTRON_WEDGE_TRACE_RUN_UUID") or uuid.uuid4().hex
        self.path = directory / f"wedge-trace-{os.getpid()}-{nonce}.ring"
        descriptor = os.open(self.path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
        try:
            os.ftruncate(descriptor, capacity * RECORD_BYTES)
            self.mapping = mmap.mmap(
                descriptor,
                capacity * RECORD_BYTES,
                access=mmap.ACCESS_WRITE,
            )
        finally:
            os.close(descriptor)

    def put(self, boundary: str, request_id: str, details: dict[str, Any]) -> None:
        sequence = self.sequence
        self.sequence += 1
        if sequence >= self.capacity:
            self.dropped += 1
            return
        try:
            task = current_task()
        except RuntimeError:
            task = None
        record = {
            "boundary": boundary,
            "campaign_uuid": os.environ.get("NEMOTRON_WEDGE_TRACE_RUN_UUID"),
            "details": details,
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "process_role": os.environ.get("NEMOTRON_WEDGE_TRACE_PROCESS_ROLE", "host"),
            "request_id": request_id,
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "task": None if task is None else task.get_name(),
            "thread_id": threading.get_ident(),
        }
        payload = json.dumps(
            record,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        if len(payload) >= RECORD_BYTES:
            details_sha256 = hashlib.sha256(
                json.dumps(details, sort_keys=True, separators=(",", ":"), default=str).encode()
            ).hexdigest()
            payload = json.dumps(
                {
                    "boundary": boundary,
                    "campaign_uuid": record["campaign_uuid"],
                    "details_sha256": details_sha256,
                    "monotonic_ns": time.monotonic_ns(),
                    "pid": os.getpid(),
                    "process_role": record["process_role"],
                    "request_id": request_id[:128],
                    "schema_version": SCHEMA_VERSION,
                    "sequence": sequence,
                    "task": record["task"],
                    "thread_id": record["thread_id"],
                    "truncated": True,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        offset = sequence * RECORD_BYTES
        self.mapping[offset : offset + RECORD_BYTES] = payload.ljust(RECORD_BYTES, b"\0")

    def records(self) -> list[dict[str, Any]]:
        count = min(self.sequence, self.capacity)
        records = []
        for index in range(count):
            offset = index * RECORD_BYTES
            raw = self.mapping[offset : offset + RECORD_BYTES].rstrip(b"\0")
            if raw:
                records.append(json.loads(raw))
        return records

    def close(self) -> None:
        self.mapping.flush()
        self.mapping.close()


def _enabled_from_env() -> bool:
    return os.environ.get("NEMOTRON_WEDGE_TRACE", "").strip().lower() in _TRUE


def _new_ring() -> _SidecarRing | None:
    if not _enabled_from_env():
        return None
    directory_text = os.environ.get("NEMOTRON_WEDGE_TRACE_DIR", "").strip()
    if not directory_text:
        raise RuntimeError("NEMOTRON_WEDGE_TRACE_DIR is required when tracing is enabled")
    capacity = int(os.environ.get("NEMOTRON_WEDGE_TRACE_CAPACITY", DEFAULT_CAPACITY))
    if capacity < 10:
        raise ValueError("NEMOTRON_WEDGE_TRACE_CAPACITY must be at least 10")
    return _SidecarRing(Path(directory_text), capacity)


_RING = _new_ring()


def trace_enabled() -> bool:
    return _RING is not None


def descriptor_key(value: Any) -> str:
    """Return a versioned digest of a canonical JSON-compatible descriptor."""
    canonical = json.dumps(
        {"schema": 1, "value": value},
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def record_boundary(boundary: str, request_id: str, **details: Any) -> None:
    """Record without blocking or allowing diagnostics to affect the caller."""
    ring = _RING
    if ring is None:
        return
    try:
        if boundary not in BOUNDARIES:
            raise ValueError(f"unknown wedge boundary: {boundary}")
        ring.put(boundary, str(request_id), details)
    except Exception:
        ring.dropped += 1


def snapshot_records() -> list[dict[str, Any]]:
    return [] if _RING is None else _RING.records()


def trace_metadata() -> dict[str, Any]:
    if _RING is None:
        return {"enabled": False}
    return {
        "capacity": _RING.capacity,
        "dropped": _RING.dropped,
        "enabled": True,
        "path": str(_RING.path),
        "records": min(_RING.sequence, _RING.capacity),
        "schema_version": SCHEMA_VERSION,
    }


def close_trace() -> None:
    global _RING
    if _RING is not None:
        _RING.close()
        _RING = None
