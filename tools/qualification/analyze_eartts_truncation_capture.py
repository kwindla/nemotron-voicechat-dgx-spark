#!/usr/bin/env python3
"""Summarize EarTTS raw/policy/decoder/PCM evidence and timing."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch


def _code(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.reshape(-1).tolist())


def _dbfs(value: Any) -> float:
    tensor = value.float()
    rms = float(torch.sqrt(torch.mean(tensor * tensor)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _steps(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "model_step":
                result.append(record)
    return result


def _nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percentile / 100) - 1]


def _timing(trace_root: Path) -> dict[str, Any]:
    values = []
    sessions = []
    for path in sorted(trace_root.glob("*/events.jsonl")):
        current = [
            float(step["server_step_ms"])
            for step in _steps(path)
            if step.get("turn_state", {}).get("rnnt", {}).get("agent_speaking")
        ]
        sessions.append({"session_id": path.parent.name, "count": len(current)})
        values.extend(current)
    return {
        "sessions": sessions,
        "count": len(values),
        "p50_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": _nearest_rank(values, 95),
        "p99_ms": _nearest_rank(values, 99),
        "max_ms": max(values),
        "over_80_ms": sum(value > 80.0 for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-trace-root", type=Path, required=True)
    parser.add_argument("--control-trace-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = [path for path in args.capture_root.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise SystemExit(f"expected one capture run, found {len(run_dirs)}")
    frame_paths = sorted(run_dirs[0].glob("frame-*.pt"))
    records = [torch.load(path, map_location="cpu", weights_only=False) for path in frame_paths]
    by_request: dict[str, dict[int, dict[str, Any]]] = {}
    for record in records:
        by_request.setdefault(str(record["request_id"]), {})[
            int(record["frame_index"])
        ] = record

    policy_records = [record for record in records if record["decoder_pad_policy_applied"]]
    silence_codes = {_code(record["decoder_code"]) for record in policy_records}
    if len(silence_codes) != 1:
        raise SystemExit(f"policy produced {len(silence_codes)} decoder code vectors")
    silence_code = next(iter(silence_codes))

    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    response_rows = []
    for attempt_index, attempt in enumerate(campaign["attempts"], start=1):
        session_id = attempt["session_id"]
        request_id = str(attempt_index + 1)  # request 1 is the startup warmup
        capture = by_request[request_id]
        steps = _steps(args.capture_trace_root / session_id / "events.jsonl")
        boundaries = [
            int(step["frame"])
            for step in steps
            if step.get("turn_state", {}).get("response_boundary", {}).get("event")
            in {"start", "end"}
        ]
        if len(boundaries) != 6:
            raise SystemExit(f"session {session_id} has boundaries {boundaries}")
        for response_index, response in enumerate(attempt["responses"]):
            start = boundaries[response_index * 2]
            end = boundaries[response_index * 2 + 1]
            selected = [capture[frame] for frame in range(start, end + 1)]
            selected_steps = steps[start : end + 1]
            raw_silence = [
                item for item in selected if _code(item["raw_generated_code"]) == silence_code
            ]
            post_silence = [
                item
                for item in selected
                if _code(item["post_policy_recurrent_code"]) == silence_code
            ]
            decoder_silence = [
                item for item in selected if _code(item["decoder_code"]) == silence_code
            ]
            pcm_dbfs = [_dbfs(item["codec_pcm"]) for item in selected]
            pcm_silence = [
                item for item, dbfs in zip(selected, pcm_dbfs, strict=True) if dbfs <= -90.0
            ]
            audible_indices = [
                index for index, dbfs in enumerate(pcm_dbfs) if dbfs > -90.0
            ]
            final_silence_index = (audible_indices[-1] + 1) if audible_indices else 0
            final_silence = selected[final_silence_index:]
            final_silence_onset = selected[final_silence_index]
            policy = [item for item in selected if item["decoder_pad_policy_applied"]]
            tail_done = [item for item in selected if item["tail_done"]]
            reached_20_at = next(
                (
                    int(step["frame"])
                    for step in selected_steps
                    if step.get("turn_state", {})
                    .get("agent_silence_watchdog", {})
                    .get("silent_frames", 0)
                    >= 20
                ),
                None,
            )
            resumed_after_20 = False
            if reached_20_at is not None:
                resumed_after_20 = any(
                    int(step["frame"]) > reached_20_at
                    and float(step["model_output_audio"]["rms_dbfs"]) > -90.0
                    for step in selected_steps
                )
            response_rows.append(
                {
                    "attempt": attempt_index,
                    "response": response_index + 1,
                    "session_id": session_id,
                    "response_id": response["response_id"],
                    "reason": response["reason"],
                    "start_frame": start,
                    "end_frame": end,
                    "positions": len(selected),
                    "raw_codec_silence_positions": len(raw_silence),
                    "post_policy_codec_silence_positions": len(post_silence),
                    "decoder_codec_silence_positions": len(decoder_silence),
                    "pcm_at_or_below_minus_90_dbfs_positions": len(pcm_silence),
                    "policy_applied_positions": len(policy),
                    "tail_done_positions": len(tail_done),
                    "first_policy_frame": (
                        int(policy[0]["frame_index"]) if policy else None
                    ),
                    "first_pcm_silence_frame": (
                        int(pcm_silence[0]["frame_index"]) if pcm_silence else None
                    ),
                    "final_pcm_silence_start_frame": int(
                        final_silence_onset["frame_index"]
                    ),
                    "final_pcm_silence_positions": len(final_silence),
                    "final_silence_onset": {
                        "pcm_dbfs": pcm_dbfs[final_silence_index],
                        "current_subword_id": int(
                            final_silence_onset["current_subword_id"]
                        ),
                        "text_control": final_silence_onset["text_control"],
                        "tts_in_turn_content": int(
                            final_silence_onset["tts_in_turn_content"]
                        ),
                        "tts_in_turn_pads": int(
                            final_silence_onset["tts_in_turn_pads"]
                        ),
                        "tail_done": bool(final_silence_onset["tail_done"]),
                        "policy_applied": bool(
                            final_silence_onset["decoder_pad_policy_applied"]
                        ),
                        "raw_equals_post_policy": torch.equal(
                            final_silence_onset["raw_generated_code"],
                            final_silence_onset["post_policy_recurrent_code"],
                        ),
                        "post_policy_equals_decoder": torch.equal(
                            final_silence_onset["post_policy_recurrent_code"],
                            final_silence_onset["decoder_code"],
                        ),
                    },
                    "reached_20_silent_frames_at": reached_20_at,
                    "audio_resumed_after_20_silent_frames": resumed_after_20,
                    "max_server_silent_frames": max(
                        int(
                            step.get("turn_state", {})
                            .get("agent_silence_watchdog", {})
                            .get("silent_frames", 0)
                        )
                        for step in selected_steps
                    ),
                }
            )

    result = {
        "schema": "eartts-truncation-acoustic-analysis-v1",
        "capture_run": run_dirs[0].name,
        "capture_positions": len(records),
        "capture_files_contiguous": [int(path.stem.split("-")[1]) for path in frame_paths]
        == list(range(len(frame_paths))),
        "codec_silence_code": list(silence_code),
        "overall": {
            "raw_codec_silence_positions": sum(
                _code(record["raw_generated_code"]) == silence_code for record in records
            ),
            "post_policy_codec_silence_positions": sum(
                _code(record["post_policy_recurrent_code"]) == silence_code
                for record in records
            ),
            "decoder_codec_silence_positions": sum(
                _code(record["decoder_code"]) == silence_code for record in records
            ),
            "policy_applied_positions": len(policy_records),
            "pcm_at_or_below_minus_90_dbfs_positions": sum(
                _dbfs(record["codec_pcm"]) <= -90.0 for record in records
            ),
            "missing_codec_pcm_positions": sum(
                record["codec_pcm"] is None for record in records
            ),
        },
        "responses": response_rows,
        "timing": {
            "control": _timing(args.control_trace_root),
            "capture": _timing(args.capture_trace_root),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
