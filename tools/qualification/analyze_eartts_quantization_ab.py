#!/usr/bin/env python3
"""Compare generated text with pinned-ASR text for the EarTTS quantization A/B."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def suffix_alignment(reference: str, hypothesis: str) -> dict[str, Any]:
    generated = normalize(reference)
    asr = normalize(hypothesis)
    matcher = difflib.SequenceMatcher(a=generated, b=asr, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    near_asr_end = [block for block in blocks if block.b + block.size >= len(asr) - 2]
    last = near_asr_end[-1] if near_asr_end else None
    suffix_start = last.a + last.size if last else 0
    suffix = generated[suffix_start:]
    matched_prefix = sum(
        block.size for block in blocks if block.a + block.size <= suffix_start
    )
    prefix_denominator = max(1, min(suffix_start, len(asr)))
    prefix_coverage = matched_prefix / prefix_denominator
    suffix_loss = bool(len(suffix) >= 3 and prefix_coverage >= 0.7)
    return {
        "generated_word_count": len(generated),
        "asr_word_count": len(asr),
        "matched_prefix_coverage": round(prefix_coverage, 4),
        "missing_suffix_word_count": len(suffix) if suffix_loss else 0,
        "missing_suffix": " ".join(suffix) if suffix_loss else None,
        "suffix_loss": suffix_loss,
    }


def load_campaign(root: Path, pair: str, arm: str) -> dict[str, Any]:
    campaign_root = root / "live" / pair / arm / "replay"
    campaign = json.loads((campaign_root / "campaign.json").read_text())
    rows = []
    for attempt_number, attempt in enumerate(campaign["attempts"], 1):
        attempt_dir = campaign_root / f"attempt-{attempt_number:02d}"
        report = json.loads((attempt_dir / "report.json").read_text())
        transcripts = json.loads((attempt_dir / "transcripts.json").read_text())
        if attempt["session_closed"]["status"] != "completed":
            raise ValueError(f"{attempt_dir}: session did not complete")
        for response in report["responses"]:
            ordinal = int(response["ordinal"])
            asr = transcripts[f"response_{ordinal:03d}.wav"]["transcript"]
            rows.append(
                {
                    "attempt": attempt_number,
                    "response": ordinal,
                    "generated_text": response["text"],
                    "asr_text": asr,
                    "audio_sha256": response["audio_sha256"],
                    **suffix_alignment(response["text"], asr),
                }
            )
    if len(rows) != 30:
        raise ValueError(f"{campaign_root}: expected 30 responses")
    return {
        "campaign_root": str(campaign_root),
        "source_pcm_sha256": campaign["source_pcm_sha256"],
        "response_count": len(rows),
        "suffix_loss_count": sum(row["suffix_loss"] for row in rows),
        "long_response_suffix_loss_count": sum(
            row["suffix_loss"] for row in rows if row["response"] == 3
        ),
        "responses": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": "nemotron_voicechat.eartts_quantization_ab_analysis.v1",
        "classification": {
            "method": "normalized-word suffix alignment",
            "minimum_missing_suffix_words": 3,
            "minimum_matched_prefix_coverage": 0.7,
            "asr_end_tolerance_words": 2,
        },
        "pairs": {},
    }
    for pair in ("pair-01", "pair-02"):
        arms = {
            arm: load_campaign(args.root, pair, arm) for arm in ("w8a32", "fp32")
        }
        generated_equal = [
            normalize(left["generated_text"]) == normalize(right["generated_text"])
            for left, right in zip(
                arms["w8a32"]["responses"], arms["fp32"]["responses"], strict=True
            )
        ]
        result["pairs"][pair] = {
            "arms": arms,
            "generated_text_equal_count": sum(generated_equal),
            "generated_text_divergent_count": len(generated_equal) - sum(generated_equal),
            "response_level_pairing_valid": all(generated_equal),
        }
    result["conclusion"] = (
        "not_paired_generated_text_diverged"
        if any(
            not pair["response_level_pairing_valid"]
            for pair in result["pairs"].values()
        )
        else "paired"
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
