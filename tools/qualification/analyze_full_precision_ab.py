#!/usr/bin/env python3
"""Analyze the long-turn harness control and full-precision lock-step arm."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def suffix_alignment(reference: str, hypothesis: str) -> dict[str, Any]:
    generated = normalize(reference)
    asr = normalize(hypothesis)
    blocks = [
        block
        for block in difflib.SequenceMatcher(
            a=generated, b=asr, autojunk=False
        ).get_matching_blocks()
        if block.size
    ]
    near_asr_end = [block for block in blocks if block.b + block.size >= len(asr) - 2]
    last = near_asr_end[-1] if near_asr_end else None
    suffix_start = last.a + last.size if last else 0
    suffix = generated[suffix_start:]
    matched_prefix = sum(
        block.size for block in blocks if block.a + block.size <= suffix_start
    )
    coverage = matched_prefix / max(1, min(suffix_start, len(asr)))
    loss = bool(len(suffix) >= 3 and coverage >= 0.7)
    return {
        "generated_word_count": len(generated),
        "asr_word_count": len(asr),
        "matched_prefix_coverage": round(coverage, 4),
        "missing_suffix_word_count": len(suffix) if loss else 0,
        "missing_suffix": " ".join(suffix) if loss else None,
        "suffix_loss": loss,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def load_arm(root: Path, arm: str, minimum_responses: int = 10) -> dict[str, Any] | None:
    campaign_root = root / "live" / arm / "replay"
    campaign_path = campaign_root / "campaign.json"
    if not campaign_path.is_file():
        return None
    campaign = json.loads(campaign_path.read_text())
    rows = []
    step_ms: list[float] = []
    processed_acks = 0
    for attempt_number, attempt in enumerate(campaign["attempts"], 1):
        attempt_dir = campaign_root / f"attempt-{attempt_number:02d}"
        transcripts = json.loads((attempt_dir / "transcripts.json").read_text())
        if attempt["session_closed"]["status"] != "completed":
            raise ValueError(f"{attempt_dir}: session did not complete")
        for response in attempt["responses"]:
            ordinal = int(response["ordinal"])
            asr_text = transcripts[f"response_{ordinal:03d}.wav"]["transcript"]
            rows.append(
                {
                    "attempt": attempt_number,
                    "response": ordinal,
                    "generated_text": response["text"],
                    "asr_text": asr_text,
                    "audio_sha256": response["audio_sha256"],
                    "termination_reason": response["reason"],
                    **suffix_alignment(response["text"], asr_text),
                }
            )
        with (attempt_dir / "events.jsonl").open() as events:
            for line in events:
                event = json.loads(line)
                if event.get("type") == "voicechat.client_event.processed":
                    processed_acks += 1
                if (
                    event.get("type") == "voicechat.metrics"
                    and event.get("frame") is not None
                    and isinstance(event.get("server_step_ms"), (int, float))
                ):
                    step_ms.append(float(event["server_step_ms"]))
    if len(rows) < minimum_responses:
        raise ValueError(
            f"{campaign_root}: expected at least {minimum_responses} story responses"
        )
    return {
        "campaign_root": str(campaign_root),
        "pacing": campaign["pacing"],
        "source_pcm_sha256": campaign["source_pcm_sha256"],
        "response_count": len(rows),
        "suffix_loss_count": sum(row["suffix_loss"] for row in rows),
        "termination_reasons": sorted({row["termination_reason"] for row in rows}),
        "processed_ack_count": processed_acks,
        "model_step_time_ms": {
            "count": len(step_ms),
            "mean": round(statistics.fmean(step_ms), 3) if step_ms else None,
            "median": round(statistics.median(step_ms), 3) if step_ms else None,
            "p95": percentile(step_ms, 0.95),
            "maximum": round(max(step_ms), 3) if step_ms else None,
            "realtime_budget_ms": 80.0,
        },
        "responses": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--control-root",
        type=Path,
        help="root retaining the quantized wall and lock-step controls",
    )
    parser.add_argument(
        "--quantization-analysis",
        type=Path,
        help="retained two-pair EarTTS quantization analysis",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control_root = args.control_root or args.root
    arms = {}
    for arm in ("quantized-wall", "quantized-lock-step"):
        if result := load_arm(control_root, arm):
            arms[arm] = result
    if result := load_arm(args.root, "full-precision-lock-step"):
        arms["full-precision-lock-step"] = result
    wall = arms.get("quantized-wall")
    lock = arms.get("quantized-lock-step")
    control_equal = bool(
        wall
        and lock
        and wall["suffix_loss_count"] == lock["suffix_loss_count"]
        and wall["source_pcm_sha256"] == lock["source_pcm_sha256"]
    )
    result = {
        "schema": "nemotron_voicechat.full_precision_ab_analysis.v1",
        "classification": {
            "method": "normalized-word suffix alignment",
            "minimum_missing_suffix_words": 3,
            "minimum_matched_prefix_coverage": 0.7,
            "asr_end_tolerance_words": 2,
        },
        "harness_control_equal_incidence": control_equal,
        "harness_interpretation": (
            "Pacing remains a bounded uncertainty; equality is not a validity gate."
        ),
        "arms": arms,
    }
    quantized_counts = []
    if args.quantization_analysis:
        prior = json.loads(args.quantization_analysis.read_text())
        for pair_name in ("pair-01", "pair-02"):
            quantized_counts.append(
                int(prior["pairs"][pair_name]["arms"]["w8a32"][
                    "long_response_suffix_loss_count"
                ])
            )
    if wall and lock:
        quantized_counts.extend(
            [int(wall["suffix_loss_count"]), int(lock["suffix_loss_count"])]
        )
    if quantized_counts:
        result["quantized_envelope"] = {
            "campaign_loss_counts": quantized_counts,
            "campaign_response_counts": [10] * len(quantized_counts),
            "pooled_loss_count": sum(quantized_counts),
            "pooled_response_count": 10 * len(quantized_counts),
            "pooled_loss_rate": sum(quantized_counts) / (10 * len(quantized_counts)),
            "observed_campaign_rate_range": [
                min(quantized_counts) / 10,
                max(quantized_counts) / 10,
            ],
        }
    if "full-precision-lock-step" in arms:
        full = arms["full-precision-lock-step"]
        losses = int(full["suffix_loss_count"])
        count = int(full["response_count"])
        p_value = sum(math.comb(count, k) for k in range(losses + 1)) / (2**count)
        loss_rate = losses / count
        if loss_rate > 0.8:
            conclusion = "quantization_exonerated_above_observed_quantized_envelope"
        elif loss_rate >= 0.2:
            conclusion = "quantization_exonerated_inside_observed_quantized_envelope"
        elif p_value <= 0.001:
            conclusion = "quantization_implicated_decisive"
        elif p_value <= 0.025:
            conclusion = "quantization_implicated_strong"
        elif p_value <= 0.1:
            conclusion = "quantization_implicated_suggestive"
        else:
            conclusion = "below_quantized_envelope_but_inconclusive"
        result["full_precision_test"] = {
            "null_loss_rate": 0.5,
            "one_sided_binomial_cdf": p_value,
            "loss_count": losses,
            "response_count": count,
            "loss_rate": loss_rate,
        }
        result["conclusion"] = conclusion
    else:
        result["conclusion"] = "full_precision_not_run"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
