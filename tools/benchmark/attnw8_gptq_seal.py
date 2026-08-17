#!/usr/bin/env python3
"""Create and verify the two hash-chained attnW8-GPTQ campaign seals."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


PLAN_SHA256 = "17647bd75dad6cd9df81aaea38a2332f76a83e73ac745c1d7e36908d86258df4"
C1_SHA256 = "1c7856088512ab3b865665682af1eaa43947ef3b1b403f6f8a13654be0a545ee"
TARGET_LAYERS = [14, 21, 30, 39]
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def tree(root: Path) -> dict[str, Any]:
    files = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {"root": str(root.resolve()), "files": files, "sha256": canonical_sha(files)}


def command(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"argv": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def read_manifest(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line]


def selected_partition(
    roots: list[Path], selection: dict[str, Any], split: str
) -> dict[str, Any]:
    records = []
    sources = []
    for chosen in selection[split]:
        index = int(chosen["root_index"])
        root = roots[index].resolve()
        ids = set(map(str, chosen["include_request_ids"]))
        chosen_records = [r for r in read_manifest(root) if str(r["request_id"]) in ids]
        for record in chosen_records:
            payload = root / record["file"]
            if payload.stat().st_size != int(record["bytes"]) or sha256(payload) != record["sha256"]:
                raise RuntimeError(f"corpus payload hash mismatch: {payload}")
        records.extend(
            {
                "source_root": str(root),
                "source_manifest_sha256": sha256(root / "manifest.jsonl"),
                "conversation_request": f"{split}:{index}:{r.get('sequence_epoch', 0)}:{r['request_id']}",
                "call_index": int(r["call_index"]),
                "payload_sha256": r["sha256"],
                "payload_bytes": int(r["bytes"]),
                "payload_file": r["file"],
            }
            for r in chosen_records
        )
        sources.append(
            {
                "root_index": index,
                "root": str(root),
                "manifest_sha256": sha256(root / "manifest.jsonl"),
                "include_request_ids": sorted(ids),
                "selected_calls": len(chosen_records),
            }
        )
    conversations = sorted({r["conversation_request"].rsplit(":", 1)[0] + ":" + r["conversation_request"].rsplit(":", 1)[1] for r in records})
    return {
        "split": split,
        "sources": sources,
        "records": records,
        "record_count": len(records),
        "conversation_count": len(conversations),
        "records_sha256": canonical_sha(records),
    }


def tool_partition(records: list[dict[str, Any]], partition: str, fixture_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    selected = [r for r in records if r["partition"] == partition]
    enriched = []
    for record in selected:
        pcm = fixture_root / record["pcm"]["path"]
        wav = fixture_root / record["wav"]["path"]
        if sha256(pcm) != record["pcm"]["sha256"] or sha256(wav) != record["wav"]["sha256"]:
            raise RuntimeError(f"fixture hash mismatch for {record['scenario_id']}")
        enriched.append(
            {
                **record,
                "resolved_pcm": str(pcm.resolve()),
                "resolved_wav": str(wav.resolve()),
                "system_prompt": payload["system_prompt"],
                "registered_tool_schema": payload["tool"],
                "injected_result": payload["injected_result"],
                "expected_call": payload["expected_call"],
            }
        )
    return {"partition": partition, "records": enriched, "count": len(enriched), "sha256": canonical_sha(enriched)}


def rules() -> dict[str, Any]:
    cells = (
        [f"layer-only-{layer}" for layer in TARGET_LAYERS]
        + [f"leave-layer-{layer}" for layer in TARGET_LAYERS]
        + [f"family-only-{p[0]}" for p in PROJECTIONS]
        + [f"leave-family-{p[0]}" for p in PROJECTIONS]
    )
    return {
        "algorithm": "sequential symmetric per-column GPTQ with Hessian error compensation",
        "target_layers": TARGET_LAYERS,
        "projection_order": PROJECTIONS,
        "bits": 8,
        "group_size": 128,
        "block_size": 128,
        "damp_percent": 0.01,
        "seed": 0,
        "dtype": {"activations": "bfloat16", "work_weights": "float32", "hessian": "float32", "damping_inverse_loss_feedback": "float32", "propagation_weight": "bfloat16", "scales_stored": "bfloat16"},
        "determinism": {"torch_deterministic_algorithms": True, "tf32": False, "float32_matmul_precision": "highest", "clean_builds": 2, "complete_artifact_and_intermediate_byte_identity": True},
        "effective_position": "prefill supplied prefix once; later calls only newly decoded row; activation r selects emitted coordinate r+1",
        "a2": {"count": 24, "positive": 12, "no_call": 12, "window_rows": 9, "positive_window": "r*-4..r*+4", "no_call_center": "first agent-BOS decision", "initiation_quota": 24},
        "scenario_pool": {
            "minimum_count": 40,
            "minimum_positive_intent": 20,
            "minimum_no_call_intent": 20,
            "order": "fixture-manifest order over a2 and c2 records",
            "selection": "first 12 validated positive-intent and first 12 validated no-call-intent in sealed pool order",
            "miss_disposition": "retain verbatim transcript, similarity, and exact categorical outcomes as propensity evidence; do not invalidate pool",
            "shortfall_disposition": "fewer than 12 validated in either class stops for adjudication without replacement loop",
            "selected_measurement_surfaces": ["a2", "c2_tool"],
        },
        "hierarchical_hessian": "H=0.75*mean_3((2/n)sum(xxT))+0.25*(0.5*mean_12_positive((2/n)sum(xxT))+0.5*mean_12_no_call((2/n)sum(xxT)))",
        "numerics": {"dead_columns": "diagonal one, offdiagonal zero, restored zero", "damping": "0.01*mean(diag(H)) on nondead block", "retry_multipliers": [1.0, 10.0, 100.0], "inverse": "float64 Cholesky inverse, symmetrize, reverse-order Cholesky", "code_range": [-127, 128], "scale": "max(absmax/127,max/128) in float64 then stored BF16 then reloaded", "qzeros": 0, "g_idx": "natural", "padding": "sealed shape-only zero padding"},
        "gates": {"C0": "all construction checks and every GPTQ/RTN cell <=1.0", "C1": "adjacent fhw8 and candidate each 12/12, exact call and args, zero fatal", "C2": "zero base replay head/conjunction flips and zero frozen tool categorical changes", "C3": {"minimum_ms_per_frame_improvement": 0.5, "max_non_target_kernel_family_regression_ms": 0.3, "baseline_ms": 50.549}, "C4": "retained full battery without new failure class"},
        "c0_accumulation": "float32 subtract/square, fixed row then feature order, sequential IEEE float64 sum; exact ZERO_RTN sentinels",
        "c2_margin": "max(function_logits)-max(text_logits); report signed absolute change and change/abs(baseline)",
        "scenario_replacement": "whole scenario only before Seal-1 via revised Seal-0 and complete affected baseline restart",
        "transcript_acceptance": {
            "version": "pooled_normalized_word_similarity_v3",
            "effective": "prospective from first Seal-0 hashing the recorded deviation",
            "normalization": "Unicode NFKC; case-fold; canonicalize explicitly hyphenated compounds across hyphenated, closed, and open renderings; remove remaining Unicode punctuation; token edit similarity with SequenceMatcher substitution costs",
            "threshold": 0.8,
            "categorical_tool_outcomes_remain_exact": True,
        },
        "fail_closed": "stop main ladder at first failed boundary; retain invalid attempts with exclusion reason",
        "contingency": {"trigger": "tool-sensitive C0 tool stratum, C1, or C2 failure", "coarse_order": cells, "within_layer_order": [f"{{layer}}-{p[0]}-{mode}" for mode in ("only", "leave") for p in PROJECTIONS], "paired_implication_rule": True, "no_early_stop_within_stage": True, "mandatory_algorithmic_control": ["base-only", "full"]},
        "c1_matrix": {"phases_in_order": ["control", "pre_response", "during_injection", "during_ack"], "cells_per_phase": 3, "cells": 12, "observe_seconds": 12.0, "between_cells_seconds": 8.0, "required_each": "exactly one get_benchmark_word call with empty arguments, no fatal/error, loop bound <=4; interrupted cells require second lifecycle"},
        "runtime_routing": {"unquantized_bf16_gemv": [8, 0], "marlin": [156, 168]},
        "baseline_capture": {
            "wrapper": "tools/conversion/nano_replay.py:install_public_nano_capture",
            "integration": "import sealed nano_replay from read-only /workspace/project/tools/conversion; wrap server.build_pipeline result before server.main",
            "capture_root_in_container": "/capture",
            "max_calls": 12000,
            "repository_mount": "/home/khkramer/src/nemotron-voicechat-dgx-spark:/workspace/project:ro",
            "only_environment_deltas": {
                "VOICECHAT_NANO_REPLAY_CAPTURE_DIR": "/capture",
                "VOICECHAT_NANO_REPLAY_MAX_CALLS": "12000",
            },
        },
    }


def seal0(args: argparse.Namespace) -> None:
    plan = args.plan.resolve()
    if sha256(plan) != PLAN_SHA256:
        raise SystemExit("plan hash differs from the triple-reviewed plan")
    plan_seal = args.plan_seal.read_text().strip()
    if PLAN_SHA256 not in plan_seal:
        raise SystemExit("recorded plan seal does not contain the plan hash")
    review = args.review_r3.read_text()
    if "## Verdict" not in review or "**PASS " not in review:
        raise SystemExit("R3 does not record PASS")
    deviation = args.criterion_deviation.read_text(encoding="utf-8")
    required_deviation_phrases = (
        "PROSPECTIVE DEVIATION",
        "Every prior invalidation stands",
        "adversarial results review",
    )
    if not all(phrase in deviation for phrase in required_deviation_phrases):
        raise SystemExit("criterion deviation record is incomplete")
    selection = json.loads(args.selection.read_text())
    fixture_manifest = json.loads(args.fixture_manifest.read_text())
    payload = json.loads(args.scenarios.read_text())
    a1 = selected_partition(args.a1_root, selection, "calibration")
    q_base = selected_partition(args.q_root, selection, "evaluation")
    a2 = tool_partition(fixture_manifest["records"], "a2", args.fixture_root, payload)
    c2 = tool_partition(fixture_manifest["records"], "c2", args.fixture_root, payload)
    c1 = tool_partition(fixture_manifest["records"], "c1", args.fixture_root, payload)
    if (a1["record_count"], a1["conversation_count"]) != (3934, 3):
        raise RuntimeError(f"A1 count mismatch: {a1['record_count']}/{a1['conversation_count']}")
    if (q_base["record_count"], q_base["conversation_count"]) != (1172, 1):
        raise RuntimeError(f"Q count mismatch: {q_base['record_count']}/{q_base['conversation_count']}")
    if (a2["count"], c2["count"], c1["count"]) != (24, 24, 1):
        raise RuntimeError("tool partition count mismatch")
    enriched_by_id = {
        record["scenario_id"]: record for record in a2["records"] + c2["records"]
    }
    pool_records = [
        enriched_by_id[record["scenario_id"]]
        for record in fixture_manifest["records"]
        if record.get("partition") in {"a2", "c2"}
    ]
    tool_pool = {
        "count": len(pool_records),
        "positive_intent": sum(record["expected"] == "positive" for record in pool_records),
        "no_call_intent": sum(record["expected"] == "no_call" for record in pool_records),
        "order": "fixture-manifest order over a2 and c2 records",
        "selection_rule": "first 12 validated positive-intent and first 12 validated no-call-intent in sealed pool order",
        "records": pool_records,
        "sha256": canonical_sha(pool_records),
    }
    if (
        tool_pool["count"] < 40
        or tool_pool["positive_intent"] < 20
        or tool_pool["no_call_intent"] < 20
    ):
        raise RuntimeError("sealed tool candidate pool does not meet minimum quotas")
    for part in (a2, c2):
        if sum(r["expected"] == "positive" for r in part["records"]) != 12:
            raise RuntimeError(f"{part['partition']} does not contain 12 positives")
        if sum(r["expected"] == "no_call" for r in part["records"]) != 12:
            raise RuntimeError(f"{part['partition']} does not contain 12 no-call controls")
    if c1["records"][0]["pcm"]["sha256"] != C1_SHA256:
        raise RuntimeError("C1 carrier hash mismatch")
    interrupt = {
        "path": str(args.c1_interrupt.resolve()),
        "bytes": args.c1_interrupt.stat().st_size,
        "sha256": sha256(args.c1_interrupt),
        "expected_transcript": "stop. what is two plus two?",
    }
    c1["records"][0]["interrupt_carrier"] = interrupt
    c1["records"][0]["matrix"] = rules()["c1_matrix"]
    c1["sha256"] = canonical_sha(c1["records"])
    tool_sets = [set(r["scenario_id"] for r in p["records"]) for p in (a2, c2, c1)]
    audio_sets = [set(r["pcm"]["sha256"] for r in p["records"]) for p in (a2, c2, c1)]
    for i in range(3):
        for j in range(i + 1, 3):
            if tool_sets[i] & tool_sets[j] or audio_sets[i] & audio_sets[j]:
                raise RuntimeError("tool scenario/audio partition collision")
    source_roots = {s["root"] for s in a1["sources"]}
    if source_roots & {s["root"] for s in q_base["sources"]}:
        raise RuntimeError("A1 and Q resolved source roots collide")
    code_paths = args.code
    code = [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in code_paths]
    inspect = command(["docker", "inspect", args.container])
    image = command(["docker", "image", "inspect", args.image])
    packages = command(["docker", "exec", args.container, "python3", "-c", "import torch,transformers,safetensors,vllm,compressed_tensors,modelopt,json; print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'safetensors':safetensors.__version__,'vllm':vllm.__version__,'compressed-tensors':compressed_tensors.__version__,'nvidia-modelopt':modelopt.__version__},sort_keys=True))"])
    gpu = command(["nvidia-smi", "-q"])
    clients = command(["ss", "-Htnp", "state", "established", "sport", "=", ":8786"])
    with urllib.request.urlopen("http://127.0.0.1:8786/health", timeout=10) as response:
        health = json.loads(response.read())
    runtime = {"container_inspect": inspect, "image_inspect": image, "packages": packages, "gpu": gpu, "kernel": platform.release(), "machine": platform.machine(), "health": health, "active_clients": clients, "quiescent": not clients["stdout"].strip()}
    if any(item["returncode"] for item in (inspect, image, packages, gpu)) or not runtime["quiescent"]:
        raise RuntimeError("runtime identity/pre-stop quiescence capture failed")
    observed_packages = json.loads(packages["stdout"].strip().splitlines()[-1])
    expected_packages = {
        "torch": "2.10.0a0+b4e4ee81d3.nv25.12",
        "transformers": "4.56.0",
        "safetensors": "0.8.0",
        "vllm": "0.17.1.dev0+gb31e9326a.fi065",
        "compressed-tensors": "0.13.0",
        "nvidia-modelopt": "0.37.0",
    }
    if observed_packages != expected_packages:
        raise RuntimeError(f"qualified package identity mismatch: {observed_packages}")
    if "NVIDIA GB10" not in gpu["stdout"] or "Driver Version" not in gpu["stdout"] or "580.142" not in gpu["stdout"]:
        raise RuntimeError("qualified GPU/driver identity mismatch")
    if platform.release() != "6.17.0-1014-nvidia" or platform.machine() not in {"aarch64", "arm64"}:
        raise RuntimeError("qualified kernel/architecture identity mismatch")
    recipe = json.loads(args.recipe.read_text())
    if recipe["corpus"]["release_sha256"] != "ad8ce2fe3a66fae72ce45703216265180066b2745dc32d829ee550b45c487bd5":
        raise RuntimeError("corpus release hash mismatch")
    output = {
        "schema": 1,
        "kind": "attnw8_gptq_seal0",
        "passed": True,
        "plan": {"path": str(plan), "sha256": sha256(plan), "recorded_seal_sha256": sha256(args.plan_seal), "r3_review_sha256": sha256(args.review_r3)},
        "partitions": {"a1": a1, "a2": a2, "q_base": q_base, "q_c2": c2, "q_c1": c1, "tool_pool": tool_pool},
        "partition_audit": {"resolved_source_roots_disjoint": True, "tool_scenario_ids_pairwise_disjoint": True, "tool_audio_sha256_pairwise_disjoint": True, "tool_payloads_complete": True},
        "inputs": {"selection": {"path": str(args.selection.resolve()), "sha256": sha256(args.selection)}, "recipe": {"path": str(args.recipe.resolve()), "sha256": sha256(args.recipe), "release_sha256": recipe["corpus"]["release_sha256"]}, "corpus_release_manifest": {"path": str(args.corpus_release.resolve()), "sha256": sha256(args.corpus_release)}, "fixture_manifest": {"path": str(args.fixture_manifest.resolve()), "sha256": sha256(args.fixture_manifest)}, "scenarios": {"path": str(args.scenarios.resolve()), "sha256": sha256(args.scenarios)}, "criterion_deviation": {"path": str(args.criterion_deviation.resolve()), "sha256": sha256(args.criterion_deviation), "prospective": True, "adversarial_review_required": True}, "fhw8": tree(args.fhw8_root), "fhw8_configs": {"config.json": sha256(args.fhw8_root / "config.json"), "quantize_config.json": sha256(args.fhw8_root / "quantize_config.json")}, "retired_rtn_numeric_comparator": tree(args.rtn_root), "model_code": tree(args.model_code_root)},
        "implementation": {"files": code, "sha256": canonical_sha(code)},
        "rules": rules(),
        "rules_sha256": canonical_sha(rules()),
        "runtime_pre_stop": runtime,
        "revision": {
            "parent_seal0_sha256": args.revision_parent,
            "reason_path": str(args.revision_reason.resolve()) if args.revision_reason else None,
            "reason_sha256": sha256(args.revision_reason) if args.revision_reason else None,
            "invalidated_evidence": args.invalidated_evidence,
        },
    }
    output["payload_sha256"] = canonical_sha(output)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    manifest = tree(args.output.parent)
    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "seal0_sha256": sha256(args.output), "payload_sha256": output["payload_sha256"]}, indent=2))


def seal1(args: argparse.Namespace) -> None:
    seal0_data = json.loads(args.seal0.read_text())
    a0 = json.loads(args.a0.read_text())
    if not seal0_data.get("passed") or not a0.get("passed"):
        raise SystemExit("Seal-1 requires passing Seal-0 and A0")
    if a0["seal0_parent_sha256"] != sha256(args.seal0):
        raise SystemExit("A0 Seal-0 parent mismatch")
    pool_selection = a0.get("pool_selection") or {}
    if (
        pool_selection.get("sufficient") is not True
        or len(pool_selection.get("selected_positive") or []) != 12
        or len(pool_selection.get("selected_no_call") or []) != 12
    ):
        raise SystemExit("Seal-1 requires a sufficient deterministic 12/12 pool selection")
    c0_manifest = {
        "reference": "exact eager fhw8, same frozen input/call state per row",
        "linear_cells": [{"layer": l, "projection": p, "strata": ["q_base", "c2_tool"]} for l in TARGET_LAYERS for p in PROJECTIONS],
        "block_cells": [{"layer": l, "strata": ["q_base", "c2_tool"]} for l in TARGET_LAYERS],
        "row_order": "exact Seal-1 order then natural output feature",
        "accumulation": rules()["c0_accumulation"],
        "gate": "GPTQ/RTN <= 1.0 in every cell; exact zero-RTN sentinel rule",
    }
    payload = {
        "schema": 1,
        "kind": "attnw8_gptq_seal1",
        "passed": True,
        "seal0_parent_sha256": sha256(args.seal0),
        "a0_sha256": sha256(args.a0),
        "final_partition_sha256": {name: part["sha256"] if "sha256" in part else part["records_sha256"] for name, part in seal0_data["partitions"].items()},
        "ordered_index_sha256": a0["ordered_index_sha256"],
        "a2_window_sha256": a0["a2_window_sha256"],
        "c2_rows_sha256": a0["c2_rows_sha256"],
        "c2_rows": a0["c2_rows"],
        "c0_measurement_manifest": c0_manifest,
        "c0_measurement_manifest_sha256": canonical_sha(c0_manifest),
        "a0_evidence_sha256": canonical_sha(a0["a0"]),
        "pool_selection": pool_selection,
        "pool_selection_sha256": canonical_sha(pool_selection),
    }
    payload["payload_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output.parent / "manifest.json").write_text(json.dumps(tree(args.output.parent), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "seal1_sha256": sha256(args.output)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    zero = sub.add_parser("seal0")
    zero.add_argument("--plan", type=Path, required=True)
    zero.add_argument("--plan-seal", type=Path, required=True)
    zero.add_argument("--review-r3", type=Path, required=True)
    zero.add_argument("--selection", type=Path, required=True)
    zero.add_argument("--recipe", type=Path, required=True)
    zero.add_argument("--corpus-release", type=Path, required=True)
    zero.add_argument("--criterion-deviation", type=Path, required=True)
    zero.add_argument("--a1-root", type=Path, action="append", required=True)
    zero.add_argument("--q-root", type=Path, action="append", required=True)
    zero.add_argument("--fixture-manifest", type=Path, required=True)
    zero.add_argument("--fixture-root", type=Path, required=True)
    zero.add_argument("--c1-interrupt", type=Path, required=True)
    zero.add_argument("--scenarios", type=Path, required=True)
    zero.add_argument("--fhw8-root", type=Path, required=True)
    zero.add_argument("--rtn-root", type=Path, required=True)
    zero.add_argument("--model-code-root", type=Path, required=True)
    zero.add_argument("--code", type=Path, action="append", required=True)
    zero.add_argument("--container", default="nemotron-voicechat-model")
    zero.add_argument("--image", required=True)
    zero.add_argument("--output", type=Path, required=True)
    zero.add_argument("--revision-parent")
    zero.add_argument("--revision-reason", type=Path)
    zero.add_argument("--invalidated-evidence", action="append", default=[])
    one = sub.add_parser("seal1")
    one.add_argument("--seal0", type=Path, required=True)
    one.add_argument("--a0", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (seal0 if args.mode == "seal0" else seal1)(args)


if __name__ == "__main__":
    main()
