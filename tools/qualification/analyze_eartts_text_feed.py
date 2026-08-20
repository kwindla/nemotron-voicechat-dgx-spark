#!/usr/bin/env python3
"""Align retained EarTTS text inputs with generated text and codec PCM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CASES = (
    {
        "name": "regular-a1-r1",
        "capture": "acoustic-capture-live",
        "analysis": "acoustic-analysis.json",
        "campaign": "captured-replay-live",
        "attempt": 1,
        "response": 1,
        "missing_suffix": "How can I help you today?",
        "asr_complete": False,
    },
    {
        "name": "regular-a1-r3",
        "capture": "acoustic-capture-live",
        "analysis": "acoustic-analysis.json",
        "campaign": "captured-replay-live",
        "attempt": 1,
        "response": 3,
        "missing_suffix": (
            "From then on, the unicorn became a symbol of hope and magic. The end."
        ),
        "asr_complete": False,
    },
    {
        "name": "regular-a3-r3",
        "capture": "acoustic-capture-live",
        "analysis": "acoustic-analysis.json",
        "campaign": "captured-replay-live",
        "attempt": 3,
        "response": 3,
        "missing_suffix": "The End.",
        "asr_complete": False,
    },
    {
        "name": "extended-a1-r3",
        "capture": "acoustic-capture-extended-live",
        "analysis": "acoustic-analysis-extended.json",
        "campaign": "captured-extended-replay-live",
        "attempt": 1,
        "response": 3,
        "missing_suffix": "The end.",
        "asr_complete": False,
    },
    {
        "name": "extended-a2-r3-complete-control",
        "capture": "acoustic-capture-extended-live",
        "analysis": "acoustic-analysis-extended.json",
        "campaign": "captured-extended-replay-live",
        "attempt": 2,
        "response": 3,
        "missing_suffix": None,
        "asr_complete": True,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dbfs(torch: Any, value: Any) -> float:
    rms = float(torch.sqrt(torch.mean(value.float() ** 2)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _word_count(text: str) -> int:
    import re

    return len(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer_json = args.tokenizer / "tokenizer.json"
    tokenizer_config = args.tokenizer / "tokenizer_config.json"
    results = []

    for case in CASES:
        attempt = int(case["attempt"])
        response = int(case["response"])
        analysis = json.loads(
            (args.diagnostic_root / str(case["analysis"])).read_text()
        )
        target = next(
            row
            for row in analysis["responses"]
            if row["attempt"] == attempt and row["response"] == response
        )
        attempt_dir = (
            args.diagnostic_root
            / str(case["campaign"])
            / f"attempt-{attempt:02}"
        )
        report = json.loads((attempt_dir / "report.json").read_text())
        generated_text = next(
            row["text"] for row in report["responses"] if row["ordinal"] == response
        )
        transcript_data = json.loads((attempt_dir / "transcripts.json").read_text())
        asr_text = transcript_data[f"response_{response:03}.wav"]["transcript"]

        capture_root = args.diagnostic_root / str(case["capture"])
        run_dirs = [path for path in capture_root.iterdir() if path.is_dir()]
        if len(run_dirs) != 1:
            raise RuntimeError(f"expected one capture run, found {len(run_dirs)}")
        start = int(target["start_frame"])
        end = int(target["end_frame"])
        request_id = str(attempt + 1)
        records = []
        for frame_path in sorted(run_dirs[0].glob("frame-*.pt")):
            record = torch.load(frame_path, map_location="cpu", weights_only=False)
            frame = int(record["frame_index"])
            if str(record["request_id"]) == request_id and start <= frame <= end:
                records.append(record)
        records.sort(key=lambda row: int(row["frame_index"]))
        if [int(row["frame_index"]) for row in records] != list(range(start, end + 1)):
            raise RuntimeError(f"{case['name']} capture is not contiguous")

        content_records = [
            row for row in records if not any(row["text_control"].values())
        ]
        delivered_ids = [int(row["current_subword_id"]) for row in content_records]
        normalized_text = generated_text.lstrip()
        reported_text_ids = [
            int(value)
            for value in tokenizer.encode(generated_text, add_special_tokens=False)
        ]
        encoded = tokenizer(
            normalized_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        required_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [list(pair) for pair in encoded["offset_mapping"]]
        exact_match = delivered_ids == required_ids
        if not exact_match:
            raise RuntimeError(f"{case['name']} delivered IDs do not match full text")
        if any(
            int(row["tts_in_turn_content_after"])
            != int(row["tts_in_turn_content"]) + 1
            for row in content_records
        ):
            raise RuntimeError(f"{case['name']} content counter did not increment by one")

        last_content = content_records[-1]
        last_content_frame = int(last_content["frame_index"])
        trailing = [row for row in records if int(row["frame_index"]) > last_content_frame]
        pad_run = []
        for row in trailing:
            if row["text_control"]["is_pad"]:
                pad_run.append(row)
            else:
                break
        after_pad = trailing[len(pad_run) :]
        if (
            not pad_run
            or any(not row["text_control"]["is_eos"] for row in after_pad)
            or len(after_pad) > 1
        ):
            raise RuntimeError(
                f"{case['name']} does not end in a contiguous PAD run and optional EOS"
            )
        pad_start = int(trailing[0]["frame_index"])
        pad_end = int(pad_run[-1]["frame_index"])
        onset = int(target["final_pcm_silence_start_frame"])
        onset_record = next(row for row in records if int(row["frame_index"]) == onset)
        post_content = [
            row for row in records if int(row["frame_index"]) > last_content_frame
        ]
        audible_post_content = [
            row for row in post_content if _dbfs(torch, row["codec_pcm"]) > -90.0
        ]
        last_audible_frame = max(
            (int(row["frame_index"]) for row in audible_post_content), default=None
        )
        samples_per_frame = int(records[0]["codec_pcm"].numel())
        frame_ms = samples_per_frame / 22050.0 * 1000.0

        missing_suffix = case["missing_suffix"]
        suffix = None
        if missing_suffix is not None:
            suffix_start = normalized_text.rfind(str(missing_suffix))
            if suffix_start < 0:
                raise RuntimeError(f"{case['name']} suffix is absent from generated text")
            suffix_token_indices = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > suffix_start and token_start < len(normalized_text)
            ]
            suffix_first_index = suffix_token_indices[0]
            suffix_last_index = suffix_token_indices[-1]
            suffix = {
                "text": missing_suffix,
                "word_count": _word_count(str(missing_suffix)),
                "char_start": suffix_start,
                "first_token_index": suffix_first_index,
                "last_token_index": suffix_last_index,
                "token_count": len(suffix_token_indices),
                "token_ids": required_ids[suffix_first_index : suffix_last_index + 1],
                "first_delivery_frame": int(
                    content_records[suffix_first_index]["frame_index"]
                ),
                "last_delivery_frame": int(
                    content_records[suffix_last_index]["frame_index"]
                ),
                "fully_delivered": suffix_last_index < len(delivered_ids),
                "delivered_before_pad_run": int(
                    content_records[suffix_last_index]["frame_index"]
                )
                < pad_start,
                "delivered_before_final_silence": int(
                    content_records[suffix_last_index]["frame_index"]
                )
                < onset,
            }

        position_rows = []
        content_index = 0
        for record in records:
            is_content = not any(record["text_control"].values())
            token_id = int(record["current_subword_id"])
            row = {
                "frame": int(record["frame_index"]),
                "token_id": token_id,
                "token_kind": (
                    "bos"
                    if record["text_control"]["is_bos"]
                    else "eos"
                    if record["text_control"]["is_eos"]
                    else "pad"
                    if record["text_control"]["is_pad"]
                    else "content"
                ),
                "content_index": content_index if is_content else None,
                "token_text": (
                    tokenizer.decode([token_id], skip_special_tokens=False)
                    if is_content
                    else None
                ),
                "subword_mask": [
                    bool(value)
                    for value in record["current_subword_mask"].reshape(-1).tolist()
                ],
                "tts_in_turn_content_before": int(record["tts_in_turn_content"]),
                "tts_in_turn_content_after": int(record["tts_in_turn_content_after"]),
                "tts_in_turn_pads_before": int(record["tts_in_turn_pads"]),
                "tts_in_turn_pads_after": int(record["tts_in_turn_pads_after"]),
                "pcm_dbfs": _dbfs(torch, record["codec_pcm"]),
                "policy_applied": bool(record["decoder_pad_policy_applied"]),
            }
            position_rows.append(row)
            if is_content:
                content_index += 1
        with (args.output / f"{case['name']}-positions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in position_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        ratio_cap = 16
        ratio_min = 5
        result = {
            "name": case["name"],
            "role": "complete_control" if case["asr_complete"] else "truncated_loss",
            "capture_run": run_dirs[0].name,
            "attempt": attempt,
            "response": response,
            "session_id": target["session_id"],
            "response_frames": {"start": start, "end": end, "positions": len(records)},
            "generated_text": generated_text,
            "normalized_generated_text": normalized_text,
            "removed_leading_whitespace_for_channel_alignment": generated_text[
                : -len(normalized_text)
            ]
            if normalized_text
            else generated_text,
            "asr_text": asr_text,
            "generated_word_count": _word_count(normalized_text),
            "asr_word_count": _word_count(asr_text),
            "tokenization": {
                "reported_text_required_content_tokens": len(reported_text_ids),
                "reported_text_token_ids": reported_text_ids,
                "reported_text_sequence_exact": delivered_ids == reported_text_ids,
                "normalized_text_required_content_tokens": len(required_ids),
                "required_content_tokens": len(required_ids),
                "delivered_content_tokens": len(delivered_ids),
                "counter_final": int(last_content["tts_in_turn_content_after"]),
                "exact_full_sequence_match": exact_match,
                "required_token_ids": required_ids,
                "delivered_token_ids": delivered_ids,
                "decoded_delivered_text": tokenizer.decode(
                    delivered_ids, skip_special_tokens=False
                ),
                "offset_mapping": offsets,
            },
            "missing_suffix": suffix,
            "delivery_timing": {
                "last_content_increment_frame": last_content_frame,
                "frames_from_last_content_to_final_silence_onset": onset
                - last_content_frame,
                "terminal_pad_start_frame": pad_start,
                "terminal_pad_end_frame": pad_end,
                "terminal_eos_frame": (
                    int(after_pad[0]["frame_index"]) if after_pad else None
                ),
                "pad_started_immediately_after_last_content": pad_start
                == last_content_frame + 1,
                "final_silence_onset_frame": onset,
                "pad_frames_before_final_silence_onset": onset - pad_start,
                "pad_counter_before_onset": int(onset_record["tts_in_turn_pads"]),
                "pcm_audible_frames_after_last_content": len(audible_post_content),
                "last_audible_pcm_frame_after_content": last_audible_frame,
                "audible_continuation_frames_to_last_audible": (
                    last_audible_frame - last_content_frame
                    if last_audible_frame is not None
                    else None
                ),
                "frame_ms": frame_ms,
            },
            "ratio_guard": {
                "ratio_cap": ratio_cap,
                "token_min": ratio_min,
                "armed": len(required_ids) >= ratio_min,
                "force_eos_threshold_talking_frames": ratio_cap * len(required_ids),
                "captured_response_positions_upper_bound": len(records),
                "could_have_fired_within_capture": len(records)
                >= ratio_cap * len(required_ids),
                "captured_eos_positions": sum(
                    int(row["text_control"]["is_eos"]) for row in records
                ),
            },
            "termination_reason": target["reason"],
            "all_subword_masks_true": all(
                all(row["subword_mask"]) for row in position_rows
            ),
        }
        results.append(result)

    summary = {
        "schema": "eartts-text-feed-analysis-v1",
        "verdict": "B_full_text_delivered_eartts_pad_continuation_became_silent",
        "tokenizer": {
            "path": str(args.tokenizer),
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "tokenizer_json_sha256": _sha256(tokenizer_json),
            "tokenizer_config_sha256": _sha256(tokenizer_config),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "cases": results,
    }
    (args.output / "analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
