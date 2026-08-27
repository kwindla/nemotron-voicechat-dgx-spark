"""Diagnostic-only asynchronous capture of the live EarTTS acoustic boundary.

The public Speech wrapper already has an opt-in capture call site guarded by
``S2S_EARTTS_ACOUSTIC_CAPTURE_DIR``.  Its baked fallback writer is deliberately
simple and performs device copies plus ``torch.save`` in the realtime loop.
This module replaces only that enabled path with a bounded, lossless producer
and a dedicated writer thread.  With the variable unset, this module is never
constructed and production execution is unchanged.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

_STOP = object()


def _materialize(value: Any) -> Any:
    """Move captured tensors to immutable CPU storage on the writer thread."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item) for item in value)
    return value


class AsyncEarTTSAcousticCapture:
    """Pair acoustic positions with codec PCM and persist them off-path."""

    def __init__(self, wrapper: Any):
        import torch

        run_dir = getattr(wrapper, "_eartts_capture_dir", None)
        if run_dir is None:
            raise RuntimeError("EarTTS acoustic capture has no initialized run directory")
        self.torch = torch
        self.wrapper = wrapper
        self.run_dir = Path(run_dir)
        self.records: queue.Queue[Any] = queue.Queue(maxsize=4096)
        self.active_call: list[dict[str, Any]] | None = None
        self.awaiting_codec: list[dict[str, Any]] = []
        self.writer_error: BaseException | None = None
        self.closed = False
        self.enqueued = 0
        self.written = 0
        self.placeholder_pcm_calls = 0
        self.writer_cpu = int(os.environ.get("S2S_EARTTS_ACOUSTIC_CAPTURE_WRITER_CPU", "10"))
        self.thread = threading.Thread(
            target=self._writer,
            name="eartts-acoustic-capture",
            daemon=True,
        )
        overlay = {
            "schema": "nemotron-live-eartts-acoustic-async-overlay-v1",
            "created_time_ns": time.time_ns(),
            "queue_capacity": self.records.maxsize,
            "writer_cpu": self.writer_cpu,
            "codec_alignment": "one_frame_pipeline_previous_submission",
            "record_schema": "nemotron-live-eartts-acoustic-frame-v2",
        }
        (self.run_dir / "async-overlay.json").write_text(
            json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.thread.start()
        atexit.register(self.close)

    def _raise_if_failed(self) -> None:
        if self.writer_error is not None:
            raise RuntimeError("EarTTS acoustic capture writer failed") from self.writer_error

    def begin_call(self) -> None:
        self._raise_if_failed()
        if self.active_call is not None:
            raise RuntimeError("EarTTS acoustic capture call nesting is unsupported")
        self.active_call = []

    def capture_frame(
        self,
        *,
        stream_id: Any,
        request_id: Any,
        frame_index: Any,
        current_subword_id: Any,
        previous_subword_id: Any,
        current_subword_mask: Any,
        raw_generated_code: Any,
        recurrent_code: Any,
        decoder_code: Any,
        agent_idle_before: Any,
        agent_idle_after: Any,
    ) -> None:
        """Buffer references only; the writer owns CPU copies and persistence."""

        del raw_generated_code  # The baked argument is downstream of the PAD policy.
        self._raise_if_failed()
        if self.active_call is None:
            raise RuntimeError("EarTTS acoustic capture frame arrived outside a model call")
        sequence = int(self.wrapper._eartts_capture_sequence)
        self.wrapper._eartts_capture_sequence = sequence + 1
        policy = dict(getattr(self.wrapper, "_eartts_acoustic_diagnostic_policy", {}) or {})
        true_raw = getattr(self.wrapper, "_eartts_acoustic_diagnostic_raw_code", None)
        if true_raw is None:
            raise RuntimeError("EarTTS capture did not retain the pre-policy generated code")
        current_token = int(current_subword_id.reshape(-1)[0].item())
        if policy.get("current_subword_id") != current_token:
            raise RuntimeError("EarTTS raw-code policy evidence belongs to another position")
        text_bos = int(self.wrapper.model.tts_model.text_bos_id)
        text_eos = int(self.wrapper.model.tts_model.text_eos_id)
        text_pad = int(self.wrapper.model.tts_model.text_pad_id)
        record = {
            "schema": "nemotron-live-eartts-acoustic-frame-v2",
            "capture_metadata_sha256": self.wrapper._eartts_capture_metadata_sha256,
            "sequence": sequence,
            "stream_id": stream_id,
            "request_id": str(request_id),
            "frame_index": int(frame_index),
            "current_subword_id": current_token,
            "previous_subword_id": previous_subword_id,
            "current_subword_mask": current_subword_mask,
            "text_control": {
                "is_bos": current_token == text_bos,
                "is_eos": current_token == text_eos,
                "is_pad": current_token == text_pad,
            },
            "raw_generated_code": true_raw,
            "post_policy_recurrent_code": recurrent_code,
            "decoder_code": decoder_code,
            "codec_pcm": None,
            "codec_pcm_alignment": None,
            "agent_idle_before": bool(agent_idle_before),
            "agent_idle_after": bool(agent_idle_after),
            "tts_in_turn_content": int(policy["prior_content_tokens"]),
            "tts_in_turn_pads": int(policy["prior_pad_tokens"]),
            "tts_in_turn_content_after": int(
                getattr(self.wrapper, "_tts_in_turn_content", 0)
            ),
            "tts_in_turn_pads_after": int(
                getattr(self.wrapper, "_tts_in_turn_pads", 0)
            ),
            "tail_ratio": float(policy["tail_ratio"]),
            "tail_done": bool(policy["tail_done"]),
            "decoder_pad_policy_applied": bool(policy["decoder_pad_policy_applied"]),
            "decoder_pad_policy_eligible": bool(policy["decoder_pad_policy_eligible"]),
        }
        self.active_call.append(record)

    def _split_pcm(self, pcm: Any, count: int) -> list[Any]:
        if count <= 0:
            return []
        samples = int(pcm.shape[-1])
        if samples % count:
            raise RuntimeError(
                f"codec PCM length {samples} is not divisible across {count} positions"
            )
        per_position = samples // count
        return [
            pcm[..., index * per_position : (index + 1) * per_position]
            for index in range(count)
        ]

    def _enqueue(self, record: dict[str, Any]) -> None:
        try:
            self.records.put_nowait(record)
        except queue.Full as exc:
            raise RuntimeError("EarTTS acoustic capture queue filled; evidence is invalid") from exc
        self.enqueued += 1

    def finish_call(self, decoded_audio_new: Any) -> None:
        self._raise_if_failed()
        if self.active_call is None:
            raise RuntimeError("EarTTS acoustic capture finished without a started call")
        current = self.active_call
        self.active_call = None
        previous = self.awaiting_codec
        if previous:
            chunks = self._split_pcm(decoded_audio_new, len(previous))
            for record, chunk in zip(previous, chunks, strict=True):
                record["codec_pcm"] = chunk
                record["codec_pcm_alignment"] = "next_wrapper_call_previous_submission"
                self._enqueue(record)
        else:
            self.placeholder_pcm_calls += 1
        self.awaiting_codec = current

    def abort_call(self, reason: str) -> None:
        if self.active_call is None:
            return
        for record in self.active_call:
            record["codec_pcm_alignment"] = f"unavailable:{reason}"
            self._enqueue(record)
        self.active_call = None

    def finalize_codec(self, finalized_audio: Any, *, samples_per_frame: int) -> None:
        """Attach the process codec's drained tail to the final submitted code."""

        self._raise_if_failed()
        if not self.awaiting_codec:
            return
        count = len(self.awaiting_codec)
        needed = samples_per_frame * count
        if int(finalized_audio.shape[-1]) < needed:
            raise RuntimeError("finalized codec audio is shorter than its pending positions")
        tail = finalized_audio[..., -needed:]
        chunks = self._split_pcm(tail, count)
        for record, chunk in zip(self.awaiting_codec, chunks, strict=True):
            record["codec_pcm"] = chunk
            record["codec_pcm_alignment"] = "finalize_codec_drained_submission"
            self._enqueue(record)
        self.awaiting_codec = []

    def _writer(self) -> None:
        try:
            try:
                os.sched_setaffinity(0, {self.writer_cpu})
            except (AttributeError, OSError):
                pass
            while True:
                record = self.records.get()
                try:
                    if record is _STOP:
                        return
                    materialized = _materialize(record)
                    sequence = int(materialized["sequence"])
                    final_path = self.run_dir / f"frame-{sequence:09d}.pt"
                    temporary_path = self.run_dir / f".frame-{sequence:09d}.pt.tmp"
                    self.torch.save(materialized, temporary_path)
                    os.replace(temporary_path, final_path)
                    self.written += 1
                finally:
                    self.records.task_done()
        except BaseException as exc:  # surfaced synchronously on the next producer call
            self.writer_error = exc

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.active_call:
            self.abort_call("capture_close_during_call")
        for record in self.awaiting_codec:
            record["codec_pcm_alignment"] = "unavailable:capture_closed_before_codec_drain"
            self._enqueue(record)
        self.awaiting_codec = []
        self.records.put(_STOP)
        self.records.join()
        self.thread.join(timeout=30)
        self._raise_if_failed()
        summary = {
            "schema": "nemotron-live-eartts-acoustic-capture-summary-v1",
            "enqueued": self.enqueued,
            "written": self.written,
            "placeholder_pcm_calls": self.placeholder_pcm_calls,
            "queue_empty": self.records.empty(),
            "writer_alive": self.thread.is_alive(),
            "writer_error": None,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def install_async_eartts_acoustic_capture(wrapper: Any) -> AsyncEarTTSAcousticCapture | None:
    """Replace the baked synchronous writer only for an explicitly enabled run."""

    if getattr(wrapper, "_eartts_capture_dir", None) is None:
        return None
    existing = getattr(wrapper, "_eartts_async_acoustic_capture", None)
    if existing is not None:
        return existing
    capture = AsyncEarTTSAcousticCapture(wrapper)
    wrapper._capture_eartts_acoustic_frame = capture.capture_frame
    wrapper._eartts_async_acoustic_capture = capture
    return capture
