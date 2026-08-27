#!/usr/bin/env python3
"""Seal-0-authorized eager fhw8 proof and effective-position index builder."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from attnw8_gptq_core import (
    AGENT_BOS_TOKEN,
    ATTENTION_LAYERS,
    PAD_TOKEN,
    Sequence,
    atomic_json,
    canonical_sha256,
    capture_inputs,
    load_fhw8_eager,
    load_sequences,
    prove_shared_qkv,
    read_capture_records,
    run_layer,
    scalar,
    selected_request_ids,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal0", type=Path, required=True)
    parser.add_argument("--fhw8-root", type=Path, required=True)
    parser.add_argument("--model-code-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path, action="append", required=True)
    parser.add_argument("--q-root", type=Path, action="append", required=True)
    parser.add_argument("--tool-capture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--live-reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalized_transcript_word_sequence(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    punctuation_free = "".join(
        character
        for character in folded
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(punctuation_free.split())


def hyphenated_compounds(value: str) -> list[tuple[str, ...]]:
    folded = unicodedata.normalize("NFKC", value).casefold()
    compounds = []
    for match in re.finditer(r"(?<!\w)(\w+(?:[\u002d\u2010-\u2015]\w+)+)(?!\w)", folded):
        parts = tuple(
            part
            for part in re.split(r"[\u002d\u2010-\u2015]+", match.group(1))
            if part
        )
        if len(parts) > 1:
            compounds.append(parts)
    return compounds


def normalized_transcript_pair(observed: str, expected: str) -> tuple[str, str]:
    observed_words = normalized_transcript_word_sequence(observed)
    expected_words = normalized_transcript_word_sequence(expected)
    compounds = set(hyphenated_compounds(observed)) | set(hyphenated_compounds(expected))
    for parts in sorted(compounds, key=lambda item: (-len(item), item)):
        joined = "".join(parts)
        open_pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in parts) + r"(?!\w)"
        observed_words = re.sub(open_pattern, joined, observed_words)
        expected_words = re.sub(open_pattern, joined, expected_words)
    return observed_words, expected_words


def transcripts_word_sequence_equal(observed: str, expected: str) -> bool:
    observed_words, expected_words = normalized_transcript_pair(observed, expected)
    return observed_words == expected_words


def transcript_word_sequence_similarity(observed: str, expected: str) -> float:
    observed_words, expected_words = normalized_transcript_pair(observed, expected)
    left = observed_words.split()
    right = expected_words.split()
    if not left and not right:
        return 1.0
    previous = [float(index) for index in range(len(right) + 1)]
    for row, left_token in enumerate(left, 1):
        current = [float(row)]
        for column, right_token in enumerate(right, 1):
            substitution = 1.0 - difflib.SequenceMatcher(
                None, left_token, right_token, autojunk=False
            ).ratio()
            current.append(
                min(
                    previous[column] + 1.0,
                    current[column - 1] + 1.0,
                    previous[column - 1] + substitution,
                )
            )
        previous = current
    similarity = 1.0 - previous[-1] / max(len(left), len(right))
    return max(0.0, min(1.0, similarity))


def event_transcript(path: Path) -> str | None:
    completed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("type") == "conversation.item.input_audio_transcription.completed":
            completed.append(str(event.get("transcript") or ""))
    return completed[-1] if completed else None


def validate_live_scenarios(
    scenarios: list[dict[str, Any]], reports: Path
) -> list[dict[str, Any]]:
    evidence = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        root = reports / scenario_id
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        transcript = event_transcript(root / "events.jsonl")
        expected_transcript = scenario["expected_transcript"]
        transcript_similarity = transcript_word_sequence_similarity(
            transcript or "", expected_transcript
        )
        transcript_ok = transcript_similarity >= 0.8
        calls = report.get("tool_calls_head") or []
        if scenario["expected"] == "positive":
            outcome_ok = (
                report.get("tool_call_count") == 1
                and len(calls) == 1
                and calls[0].get("name") == "get_benchmark_word"
                and json.loads(calls[0].get("arguments") or "null") == {}
                and report.get("injections_sent") == 1
                and report.get("session_fatal") is None
                and not report.get("errors")
            )
        else:
            outcome_ok = (
                report.get("tool_call_count") == 0
                and len(calls) == 0
                and report.get("injections_sent") == 0
                and report.get("session_fatal") is None
                and not report.get("errors")
            )
        item = {
            "scenario_id": scenario_id,
            "expected": scenario["expected"],
            "expected_transcript": expected_transcript,
            "observed_transcript": transcript,
            "normalized_word_sequence_similarity": transcript_similarity,
            "transcript_similarity_threshold": 0.8,
            "transcript_acceptance": "pooled_normalized_word_similarity_v3",
            "categorical_outcome_ok": outcome_ok,
            "candidate_validated": transcript_ok and outcome_ok,
            "report_sha256": sha256_file(root / "report.json"),
            "events_sha256": sha256_file(root / "events.jsonl"),
        }
        evidence.append(item)
    return evidence


def eager_heads(hidden: list[Any], model: Any, function_weight: Any, torch):
    text_tokens: list[list[int]] = []
    function_tokens: list[list[int]] = []
    with torch.inference_mode():
        for sequence in hidden:
            normalized = model.backbone.norm_f(sequence)
            text_parts = []
            function_parts = []
            for start in range(0, normalized.shape[1], 128):
                rows = normalized[:, start : start + 128]
                text_parts.extend(model.lm_head(rows).argmax(dim=-1).reshape(-1).cpu().tolist())
                function_parts.extend(
                    torch.nn.functional.linear(rows, function_weight)
                    .argmax(dim=-1)
                    .reshape(-1)
                    .cpu()
                    .tolist()
                )
            text_tokens.append([int(value) for value in text_parts])
            function_tokens.append([int(value) for value in function_parts])
    return text_tokens, function_tokens


def propagate_all(model: Any, sequences: list[Sequence], torch, device: str):
    hidden = [sequence.embeds.to(device=device, dtype=torch.bfloat16) for sequence in sequences]
    for layer in model.backbone.layers:
        hidden = run_layer(layer, hidden, torch)
    return hidden


def build_index(
    label: str,
    sequences: list[Sequence],
    text_tokens: list[list[int]],
    function_tokens: list[list[int]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    flat = []
    by_sequence = []
    for sequence_ordinal, sequence in enumerate(sequences):
        if not (
            len(sequence.records)
            == len(text_tokens[sequence_ordinal])
            == len(function_tokens[sequence_ordinal])
            == int(sequence.embeds.shape[1])
        ):
            raise RuntimeError(f"indexed/captured row-count mismatch for {sequence.key}")
        rows = []
        for causal_row, (record, text_token, function_token) in enumerate(
            zip(
                sequence.records,
                text_tokens[sequence_ordinal],
                function_tokens[sequence_ordinal],
                strict=True,
            )
        ):
            phase = (
                "function"
                if function_token != PAD_TOKEN
                else "text"
                if text_token != PAD_TOKEN
                else "idle"
            )
            canonical = {
                "partition": label,
                "source_root": sequence.source_root,
                "source_manifest_sha256": sha256_file(
                    Path(sequence.source_root) / "manifest.jsonl"
                ),
                "conversation_request": sequence.request_id,
                "turn": 0,
                "nano_call": int(record["call_index"]),
                "causal_row": causal_row,
                "emitted_text_token": text_token,
                "emitted_function_token": function_token,
                "phase": phase,
                "scenario_id": sequence.scenario_id,
                "audio_sha256": sequence.audio_sha256,
            }
            item = {**canonical, "record_sha256": canonical_sha256(canonical)}
            rows.append(item)
            flat.append(item)
        by_sequence.append(rows)
    keys = [item["record_sha256"] for item in flat]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{label}: duplicate canonical effective-position records")
    return flat, by_sequence


def tool_windows(
    scenarios: list[dict[str, Any]],
    rows: list[list[dict[str, Any]]],
    *,
    partition: str,
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    windows = {}
    c2_rows = []
    for scenario, sequence_rows in zip(scenarios, rows, strict=True):
        function_positions = [
            i for i, row in enumerate(sequence_rows) if row["emitted_function_token"] != PAD_TOKEN
        ]
        bos_positions = [
            i for i, row in enumerate(sequence_rows) if row["emitted_text_token"] == AGENT_BOS_TOKEN
        ]
        if scenario["expected"] == "positive":
            if not function_positions:
                raise RuntimeError(f"{scenario['scenario_id']}: no eager first function token")
            center = function_positions[0]
        else:
            if function_positions:
                raise RuntimeError(f"{scenario['scenario_id']}: eager unexpected function token")
            if not bos_positions:
                raise RuntimeError(f"{scenario['scenario_id']}: no eager agent-BOS center")
            center = bos_positions[0]
        if center < 4 or center + 4 >= len(sequence_rows):
            raise RuntimeError(f"{scenario['scenario_id']}: required nine-row window unavailable")
        window = list(range(center - 4, center + 5))
        windows[scenario["scenario_id"]] = window
        if partition == "c2":
            for coordinate in range(center - 4, len(sequence_rows)):
                row = sequence_rows[coordinate]
                c2_rows.append(
                    {
                        "canonical_row_key": row["record_sha256"],
                        "scenario_id": scenario["scenario_id"],
                        "scenario_type": scenario["expected"],
                        "role": "boundary" if coordinate in window else "continuation",
                        "causal_coordinate": coordinate,
                        "nano_call": row["nano_call"],
                        "fhw8_emitted_text_token": row["emitted_text_token"],
                        "fhw8_emitted_function_token": row["emitted_function_token"],
                        "expected_tool": (
                            {"name": "get_benchmark_word", "arguments": {}}
                            if scenario["expected"] == "positive"
                            else None
                        ),
                        "extra_call_count": 0,
                        "post_result_continuation": scenario["expected"] == "positive",
                        "loop_bound": 4,
                    }
                )
    return windows, c2_rows


def main() -> None:
    args = parse_args()
    seal0 = json.loads(args.seal0.read_text(encoding="utf-8"))
    if seal0.get("kind") != "attnw8_gptq_seal0" or not seal0.get("passed"):
        raise SystemExit("A0 requires a passing Seal-0")
    if (
        seal0.get("rules", {}).get("transcript_acceptance", {}).get("version")
        != "pooled_normalized_word_similarity_v3"
    ):
        raise SystemExit("A0 requires the prospectively amended transcript criterion")
    fixtures = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))["records"]
    tool_scenarios = [item for item in fixtures if item["partition"] in {"a2", "c2"}]
    sealed_pool_ids = [
        item["scenario_id"]
        for item in seal0.get("partitions", {}).get("tool_pool", {}).get("records", [])
    ]
    if sealed_pool_ids != [item["scenario_id"] for item in tool_scenarios]:
        raise SystemExit("fixture pool order differs from the Seal-0 pool")
    live_evidence = validate_live_scenarios(tool_scenarios, args.live_reports)
    evidence_by_id = {item["scenario_id"]: item for item in live_evidence}
    selected_positive = [
        item["scenario_id"]
        for item in tool_scenarios
        if item["expected"] == "positive"
        and evidence_by_id[item["scenario_id"]]["candidate_validated"]
    ][:12]
    selected_no_call = [
        item["scenario_id"]
        for item in tool_scenarios
        if item["expected"] == "no_call"
        and evidence_by_id[item["scenario_id"]]["candidate_validated"]
    ][:12]
    selected_id_set = set(selected_positive + selected_no_call)
    selected_ids = [
        item["scenario_id"] for item in tool_scenarios if item["scenario_id"] in selected_id_set
    ]
    pool_selection = {
        "rule": "first 12 validated positive-intent and first 12 validated no-call-intent in sealed fixture-manifest pool order",
        "pool_count": len(tool_scenarios),
        "positive_intent_count": sum(item["expected"] == "positive" for item in tool_scenarios),
        "no_call_intent_count": sum(item["expected"] == "no_call" for item in tool_scenarios),
        "validated_positive_count": sum(
            item["expected"] == "positive"
            and evidence_by_id[item["scenario_id"]]["candidate_validated"]
            for item in tool_scenarios
        ),
        "validated_no_call_count": sum(
            item["expected"] == "no_call"
            and evidence_by_id[item["scenario_id"]]["candidate_validated"]
            for item in tool_scenarios
        ),
        "selected_positive": selected_positive,
        "selected_no_call": selected_no_call,
        "selected_in_pool_order": selected_ids,
        "sufficient": len(selected_positive) == 12 and len(selected_no_call) == 12,
    }
    if not pool_selection["sufficient"]:
        finding = {
            "schema": 1,
            "kind": "attnw8_gptq_a0_pool_shortfall",
            "seal0_parent_sha256": sha256_file(args.seal0),
            "a0": {"live_scenario_evidence": live_evidence},
            "pool_selection": pool_selection,
            "passed": False,
            "disposition": "stop_for_adjudication_no_scenario_replacement_loop",
        }
        atomic_json(args.output, finding)
        raise SystemExit("pooled baseline has fewer than 12 validated candidates in an intent class")

    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    a1 = load_sequences(
        args.a1_root,
        torch=torch,
        include_request_ids=selected_request_ids(selection, "calibration"),
    )
    q = load_sequences(
        args.q_root,
        torch=torch,
        include_request_ids=selected_request_ids(selection, "evaluation"),
    )
    all_tool = load_sequences([args.tool_capture], torch=torch)
    if len(all_tool) < len(tool_scenarios):
        raise RuntimeError(
            f"tool capture has only {len(all_tool)} sequences for {len(tool_scenarios)} pool scenarios"
        )
    # Capture request -1 is excluded by load_sequences, while the public server
    # warmup leaves one additional non--1 sequence ahead of the pool. The pool
    # is therefore the final N sequences, in the sealed runner order.
    all_tool = all_tool[-len(tool_scenarios) :]
    for sequence, scenario in zip(all_tool, tool_scenarios, strict=True):
        sequence.scenario_id = scenario["scenario_id"]
        sequence.audio_sha256 = scenario["pcm"]["sha256"]
    selected_pairs = [
        (scenario, sequence)
        for scenario, sequence in zip(tool_scenarios, all_tool, strict=True)
        if scenario["scenario_id"] in selected_id_set
    ]
    a2_scenarios = [scenario for scenario, _sequence in selected_pairs]
    c2_scenarios = a2_scenarios
    a2 = [sequence for _scenario, sequence in selected_pairs]
    c2 = a2
    if sum(len(item.records) for item in a1) != 3934 or len(a1) != 3:
        raise RuntimeError("A1 is not the sealed three-conversation, 3,934-call selection")
    if sum(len(item.records) for item in q) != 1172 or len(q) != 1:
        raise RuntimeError("Q is not the sealed one-conversation, 1,172-call selection")

    model, function_weight, _reader = load_fhw8_eager(
        args.fhw8_root, args.model_code_root, device=args.device, torch=torch
    )
    preflight = a2[0]
    preflight_inputs = [preflight.embeds.to(args.device, dtype=torch.bfloat16)]
    plain = preflight_inputs
    for layer in model.backbone.layers:
        plain = run_layer(layer, plain, torch)
    hooked = preflight_inputs
    qkv_proofs = []
    actual_o_sites = []
    for layer_index, layer in enumerate(model.backbone.layers):
        if layer_index in ATTENTION_LAYERS:
            outputs, captures = capture_inputs(
                layer, hooked, ("q_proj", "k_proj", "v_proj", "o_proj"), torch
            )
            prove_shared_qkv(captures, torch)
            qkv_proofs.append(
                {
                    "layer": layer_index,
                    "element_count": int(captures["q_proj"][0].numel()),
                    "q_equals_k": torch.equal(captures["q_proj"][0], captures["k_proj"][0]),
                    "q_equals_v": torch.equal(captures["q_proj"][0], captures["v_proj"][0]),
                }
            )
            actual_o_sites.append(
                {
                    "layer": layer_index,
                    "shape": list(captures["o_proj"][0].shape),
                    "sha256": __import__("hashlib").sha256(
                        captures["o_proj"][0]
                        .to(torch.float32)
                        .cpu()
                        .contiguous()
                        .numpy()
                        .tobytes()
                    ).hexdigest(),
                }
            )
            hooked = outputs
        else:
            hooked = run_layer(layer, hooked, torch)
    hook_exact = torch.equal(plain[0], hooked[0])
    if not hook_exact:
        raise RuntimeError("A0 hook transparency failed on final hidden state")

    combined = a1 + a2 + q + c2
    final_hidden = propagate_all(model, combined, torch, args.device)
    text, function = eager_heads(final_hidden, model, function_weight, torch)
    splits = [len(a1), len(a2), len(q), len(c2)]
    cursor = 0
    indexes = {}
    indexed_by_sequence = {}
    for label, count in zip(("a1", "a2", "q_base", "c2_tool"), splits, strict=True):
        flat, grouped = build_index(
            label,
            combined[cursor : cursor + count],
            text[cursor : cursor + count],
            function[cursor : cursor + count],
        )
        indexes[label] = flat
        indexed_by_sequence[label] = grouped
        cursor += count
    a2_windows, _ = tool_windows(
        a2_scenarios, indexed_by_sequence["a2"], partition="a2"
    )
    _c2_windows, c2_rows = tool_windows(
        c2_scenarios, indexed_by_sequence["c2_tool"], partition="c2"
    )
    output = {
        "schema": 1,
        "kind": "attnw8_gptq_a0_and_effective_indexes",
        "seal0_parent_sha256": sha256_file(args.seal0),
        "fhw8_root": str(args.fhw8_root.resolve()),
        "counts": {
            key: len(value) for key, value in indexes.items()
        },
        "a0": {
            "eager_reconstruction": True,
            "preflight_scenario": preflight.scenario_id,
            "hook_final_hidden_bitwise_equal": hook_exact,
            "qkv_shared_stream_proofs": qkv_proofs,
            "actual_o_input_sites": actual_o_sites,
            "live_scenario_evidence": live_evidence,
        },
        "pool_selection": pool_selection,
        "indexes": indexes,
        "ordered_index_sha256": {
            key: canonical_sha256(value) for key, value in indexes.items()
        },
        "a2_windows": a2_windows,
        "a2_window_sha256": canonical_sha256(a2_windows),
        "c2_rows": c2_rows,
        "c2_rows_sha256": canonical_sha256(c2_rows),
        "passed": True,
    }
    atomic_json(args.output, output)
    print(json.dumps({"counts": output["counts"], "passed": True}, indent=2))


if __name__ == "__main__":
    main()
