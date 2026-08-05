"""Safe local IPC for the isolated Pocket TTS worker.

Messages are bounded, length-prefixed UTF-8 JSON. Audio is returned as a
separate bounded length-prefixed byte frame. The protocol intentionally never
uses pickle or Python object serialization.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

MAX_JSON_BYTES = 64 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024


async def read_frame(reader: asyncio.StreamReader, *, limit: int) -> bytes:
    """Read one bounded big-endian length-prefixed frame."""

    size = struct.unpack(">I", await reader.readexactly(4))[0]
    if size > limit:
        raise ValueError(f"IPC frame exceeds {limit} bytes")
    return await reader.readexactly(size)


async def write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    """Write one length-prefixed frame and flush it."""

    writer.write(struct.pack(">I", len(payload)))
    writer.write(payload)
    await writer.drain()


async def read_json(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Decode one bounded JSON object."""

    payload = await read_frame(reader, limit=MAX_JSON_BYTES)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("IPC JSON payload must be an object")
    return value


async def write_json(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
    """Encode one compact JSON object."""

    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("IPC JSON payload is too large")
    await write_frame(writer, payload)

