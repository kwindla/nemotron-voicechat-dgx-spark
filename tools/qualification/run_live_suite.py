#!/usr/bin/env python3
"""Orchestrate the DGX-only restart, browser, voice, sustained, and ASR gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from nemotron_voicechat_runtime.artifacts import atomic_json, load_config
from nemotron_voicechat_runtime.bootstrap_download import extract_replay_corpus
from nemotron_voicechat_runtime.provenance import valid_runtime_image_id
from nemotron_voicechat_runtime.semantic_corpus import corpus_sha256, load_semantic_corpus
from nemotron_voicechat_runtime.tool_freshness_contract import scenario_document
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    canonical_sha256 as tool_freshness_canonical_sha256,
)
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    validate_fixture_source_provenance,
    validate_tool_freshness_fixtures,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_ASR_EVALUATOR_IMAGE_ID = (
    "sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d"
)

FP32_QUALIFICATION_COMPONENT_PATHS = {
    "nano": {
        "source": "/work/fp32/nano-vllm-fp32",
        "container": "/derived/nano-vllm-fp32",
    },
    "eartts": {
        "source": "/work/fp32/eartts-vllm-fp32",
        "container": "/derived/eartts-vllm-fp32",
    },
}

DIRECT_SEMANTIC_TOKEN_FORMS = ("text", "user_bos_text_user_eos")
DIRECT_SEMANTIC_STRUCTURAL_CHECKS = (
    "nonempty",
    "bf16_source_1x1x4480",
    "effective_outputs_forced_pad",
    "hidden_audio_discarded",
    "clocks_advance_once",
    "non_source_state_held",
    "feedback_is_exact_pad_chain",
    "perception_held_during_direct",
    "audio_clock_held_during_direct",
    "eartts_held_during_direct",
    "first_settlement_pcm_initialized_perception",
    "settlement_reached_blank_fence",
    "settlement_within_bound",
    "settlement_pre_eou_clean",
    "eartts_held_during_settlement",
    "eou_boundary_position_exactly_once",
    "eartts_bos_reset_exactly_once",
    "response_started",
    "response_complete",
    "response_audio_nonempty",
)

DIRECT_SEMANTIC_SETTLEMENT_KEYS = (
    "target_blank_frames",
    "starting_blank_frames",
    "ending_blank_frames",
    "max_model_steps",
    "model_steps",
    "eartts_reset_count_starting",
    "fence_reached",
    "within_bound",
    "pre_eou_clean",
    "steps",
)
DIRECT_SEMANTIC_SETTLEMENT_STEP_KEYS = (
    "model_step",
    "frame_index",
    "audio_frame_index_after",
    "perception_frame_idx_after",
    "eartts_reset_count_after",
    "rnnt_blank_frames",
    "agent_control",
    "response_boundary",
    "response_audio_deliverable",
    "assistant_delta",
    "assistant_text_unchanged",
    "function_delta",
    "function_text_unchanged",
    "function_cycle_effective",
    "function_calling",
    "vllm_request_positions",
    "violations",
)
DIRECT_SEMANTIC_EOU_BOUNDARY_KEYS = (
    "performed",
    "audio_frame_index_before",
    "audio_frame_index_after",
    "perception_frame_idx_before",
    "perception_frame_idx_after",
    "frame_index",
    "eartts_reset_count_after",
    "agent_control",
    "response_boundary",
    "vllm_request_positions",
    "sotc_before",
    "sotc_after",
    "sotc_final",
)
DIRECT_SEMANTIC_SOTC_KEYS = (
    "pre_eou_suppressed_tokens",
    "pre_eou_last_raw_token_id",
    "pre_eou_suppressed_frame",
    "client_eou_sotc_committed_count",
    "client_eou_sotc_committed_frame",
    "post_fc_client_bos_pending",
    "post_fc_client_bos_requested_frame",
    "post_fc_client_bos_forced_count",
    "post_fc_client_bos_forced_frame",
)
DIRECT_SEMANTIC_FUNCTION_STATE_KEYS = (
    "active",
    "awaiting_response",
    "injecting_response",
    "forced_tokens",
    "background_active",
    *DIRECT_SEMANTIC_SOTC_KEYS,
)
DIRECT_SEMANTIC_FUNCTION_TRACE_PHASES = (
    "direct_position",
    "settlement",
    "eou_boundary",
    "post_bos",
)
DIRECT_SEMANTIC_FUNCTION_TRACE_WATCHED = (
    "pad",
    "sotc",
    "eotc",
    "eotr",
    "text_bos",
)
DIRECT_SEMANTIC_FUNCTION_TRACE_ENTRY_KEYS = (
    "top_k_requested",
    "vocab_size",
    "head_argmax_id",
    "head_argmax_logit",
    "head_argmax_equal_count",
    "predicted_token_id",
    "argmax_matches_predicted",
    "sotc_rank",
    "sotc_equal_count",
    "watched",
    "top_k",
    "model_frame",
    "raw_function_token_id",
    "effective_function_token_id",
    "effective_token_reason",
    "phase",
    "phase_position",
)


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_function_logit_trace(
    case: dict[str, Any],
    label: str,
    *,
    top_k: int,
    post_bos_frames: int,
    watched_token_ids: dict[str, int],
) -> None:
    entries = case.get("function_logit_trace")
    if not isinstance(entries, list):
        raise RuntimeError(f"{label} has no function-logit trace")
    direct_count = len(case["token_ids"])
    settlement_count = case["settlement"]["model_steps"]
    boundary_count = int(case["eou_boundary"]["performed"])
    required_prefix = (
        [("direct_position", index) for index in range(direct_count)]
        + [("settlement", index) for index in range(settlement_count)]
        + [("eou_boundary", 0)] * boundary_count
    )
    if len(entries) < len(required_prefix):
        raise RuntimeError(f"{label} has a partial function-logit phase trace")
    post_count = len(entries) - len(required_prefix)
    if post_count < 0 or post_count > post_bos_frames or (post_count and not boundary_count):
        raise RuntimeError(f"{label} has an invalid post-BOS trace length")
    expected_phases = required_prefix + [("post_bos", index) for index in range(post_count)]
    actual_phases = [
        (entry.get("phase"), entry.get("phase_position"))
        if isinstance(entry, dict)
        else (None, None)
        for entry in entries
    ]
    if actual_phases != expected_phases:
        raise RuntimeError(f"{label} has ambiguous function-logit phase coverage")

    model_frames: list[int] = []
    for entry in entries:
        if set(entry) != set(DIRECT_SEMANTIC_FUNCTION_TRACE_ENTRY_KEYS):
            raise RuntimeError(f"{label} has an invalid function-logit entry shape")
        integer_keys = (
            "top_k_requested",
            "vocab_size",
            "head_argmax_id",
            "head_argmax_equal_count",
            "predicted_token_id",
            "sotc_rank",
            "sotc_equal_count",
            "model_frame",
            "raw_function_token_id",
            "effective_function_token_id",
            "phase_position",
        )
        if any(not _is_int(entry[key]) for key in integer_keys):
            raise RuntimeError(f"{label} has non-integer function-logit evidence")
        if entry["top_k_requested"] != top_k or entry["vocab_size"] < top_k:
            raise RuntimeError(f"{label} has inconsistent function-logit dimensions")
        vocab_size = entry["vocab_size"]
        token_keys = (
            "head_argmax_id",
            "predicted_token_id",
            "raw_function_token_id",
            "effective_function_token_id",
        )
        if any(not 0 <= entry[key] < vocab_size for key in token_keys):
            raise RuntimeError(f"{label} has an out-of-range function token")
        if (
            entry["argmax_matches_predicted"] is not True
            or entry["head_argmax_id"] != entry["predicted_token_id"]
            or entry["predicted_token_id"] != entry["raw_function_token_id"]
        ):
            raise RuntimeError(f"{label} contradicts the greedy function-head output")
        if entry["sotc_rank"] < 1 or entry["sotc_rank"] > vocab_size:
            raise RuntimeError(f"{label} has an invalid SOTC rank")
        if entry["sotc_equal_count"] < 1:
            raise RuntimeError(f"{label} has an invalid SOTC tie count")
        if entry["sotc_rank"] + entry["sotc_equal_count"] - 1 > vocab_size:
            raise RuntimeError(f"{label} has an impossible SOTC rank interval")
        if (
            entry["head_argmax_equal_count"] < 1
            or entry["head_argmax_equal_count"] > vocab_size
            or not isinstance(entry["head_argmax_logit"], float)
            or not math.isfinite(entry["head_argmax_logit"])
        ):
            raise RuntimeError(f"{label} has invalid function argmax tie evidence")
        if entry["phase"] not in DIRECT_SEMANTIC_FUNCTION_TRACE_PHASES:
            raise RuntimeError(f"{label} has an invalid function-logit phase")
        reason = entry["effective_token_reason"]
        if reason not in {
            "model",
            "direct_hidden_pad",
            "pre_eou_guard",
            "boundary_sotc_commit",
            "function_state_machine",
        }:
            raise RuntimeError(f"{label} has an invalid effective-token reason")
        tokens_differ = entry["raw_function_token_id"] != entry["effective_function_token_id"]
        if (
            (tokens_differ and reason in {"model", "boundary_sotc_commit"})
            or (not tokens_differ and reason not in {"model", "boundary_sotc_commit"})
            or (reason == "direct_hidden_pad" and entry["phase"] != "direct_position")
            or (reason == "boundary_sotc_commit" and entry["phase"] != "eou_boundary")
            or (
                reason in {"direct_hidden_pad", "pre_eou_guard"}
                and entry["effective_function_token_id"] != watched_token_ids["pad"]
            )
            or (
                reason == "boundary_sotc_commit"
                and entry["effective_function_token_id"] != watched_token_ids["sotc"]
            )
        ):
            raise RuntimeError(f"{label} has unexplained raw/effective function tokens")

        watched = entry["watched"]
        if not isinstance(watched, dict) or set(watched) != set(
            DIRECT_SEMANTIC_FUNCTION_TRACE_WATCHED
        ):
            raise RuntimeError(f"{label} has incomplete watched function logits")
        for name, expected_id in watched_token_ids.items():
            item = watched[name]
            if (
                not isinstance(item, dict)
                or set(item) != {"token_id", "logit"}
                or item["token_id"] != expected_id
                or not isinstance(item["logit"], float)
                or not math.isfinite(item["logit"])
            ):
                raise RuntimeError(f"{label} has invalid watched function logits")

        ranked = entry["top_k"]
        if not isinstance(ranked, list) or len(ranked) != top_k:
            raise RuntimeError(f"{label} has incomplete function top-k evidence")
        last_logit = math.inf
        ranked_ids: list[int] = []
        for item in ranked:
            if (
                not isinstance(item, dict)
                or set(item) != {"token_id", "logit"}
                or not _is_int(item["token_id"])
                or not 0 <= item["token_id"] < vocab_size
                or not isinstance(item["logit"], float)
                or not math.isfinite(item["logit"])
                or item["logit"] > last_logit
            ):
                raise RuntimeError(f"{label} has invalid function top-k evidence")
            ranked_ids.append(item["token_id"])
            last_logit = item["logit"]
        if (
            len(set(ranked_ids)) != len(ranked_ids)
            or ranked[0]["logit"] != entry["head_argmax_logit"]
        ):
            raise RuntimeError(f"{label} has contradictory function top-k evidence")
        if entry["head_argmax_id"] in ranked_ids:
            argmax_position = ranked_ids.index(entry["head_argmax_id"]) + 1
            if argmax_position > entry["head_argmax_equal_count"]:
                raise RuntimeError(f"{label} has contradictory function argmax tie evidence")
        elif entry["head_argmax_equal_count"] <= top_k:
            raise RuntimeError(f"{label} omitted an in-range function argmax from top-k")
        watched_by_id = {item["token_id"]: item["logit"] for item in watched.values()}
        if any(
            item["token_id"] in watched_by_id and item["logit"] != watched_by_id[item["token_id"]]
            for item in ranked
        ):
            raise RuntimeError(f"{label} contradicts watched and top-k logits")
        sotc_id = watched_token_ids["sotc"]
        if sotc_id in ranked_ids:
            sotc_top_k_position = ranked_ids.index(sotc_id) + 1
            if not (
                entry["sotc_rank"]
                <= sotc_top_k_position
                < entry["sotc_rank"] + entry["sotc_equal_count"]
            ):
                raise RuntimeError(f"{label} has contradictory SOTC rank evidence")
        elif entry["sotc_rank"] + entry["sotc_equal_count"] - 1 <= top_k:
            raise RuntimeError(f"{label} omitted an in-range SOTC from top-k")
        model_frames.append(entry["model_frame"])
    if model_frames and model_frames != list(
        range(model_frames[0], model_frames[0] + len(model_frames))
    ):
        raise RuntimeError(f"{label} has duplicate or discontinuous trace frames")


def _validate_sotc_evidence(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != set(DIRECT_SEMANTIC_SOTC_KEYS):
        return False
    if not _is_bool(value["post_fc_client_bos_pending"]):
        return False
    for key in (
        "pre_eou_suppressed_tokens",
        "client_eou_sotc_committed_count",
        "post_fc_client_bos_forced_count",
    ):
        if not _is_int(value[key]) or value[key] < 0:
            return False
    return all(
        value[key] is None or _is_int(value[key])
        for key in (
            "pre_eou_last_raw_token_id",
            "pre_eou_suppressed_frame",
            "client_eou_sotc_committed_frame",
            "post_fc_client_bos_requested_frame",
            "post_fc_client_bos_forced_frame",
        )
    )


def _validate_direct_settlement_evidence(case: dict[str, Any], label: str) -> None:
    settlement = case.get("settlement")
    if not isinstance(settlement, dict) or set(settlement) != set(DIRECT_SEMANTIC_SETTLEMENT_KEYS):
        raise RuntimeError(f"{label} has incomplete settlement evidence")
    for key in (
        "target_blank_frames",
        "starting_blank_frames",
        "ending_blank_frames",
        "max_model_steps",
        "model_steps",
        "eartts_reset_count_starting",
    ):
        if not _is_int(settlement[key]) or settlement[key] < 0:
            raise RuntimeError(f"{label} has invalid settlement counts")
    if settlement["target_blank_frames"] < 1 or settlement["max_model_steps"] != (
        settlement["target_blank_frames"] * 2
    ):
        raise RuntimeError(f"{label} has an inconsistent settlement bound")
    for key in ("fence_reached", "within_bound", "pre_eou_clean"):
        if not _is_bool(settlement[key]):
            raise RuntimeError(f"{label} has invalid settlement verdicts")
    steps = settlement["steps"]
    if not isinstance(steps, list) or len(steps) != settlement["model_steps"]:
        raise RuntimeError(f"{label} has inconsistent settlement steps")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or set(step) != set(DIRECT_SEMANTIC_SETTLEMENT_STEP_KEYS):
            raise RuntimeError(f"{label} has incomplete settlement step evidence")
        for key in (
            "model_step",
            "frame_index",
            "audio_frame_index_after",
            "perception_frame_idx_after",
            "eartts_reset_count_after",
            "rnnt_blank_frames",
        ):
            if not _is_int(step[key]):
                raise RuntimeError(f"{label} has invalid settlement step counters")
        if step["model_step"] != index:
            raise RuntimeError(f"{label} has inconsistent settlement step order")
        if step["agent_control"] is not None and not isinstance(step["agent_control"], str):
            raise RuntimeError(f"{label} has invalid settlement agent control")
        if not isinstance(step["response_boundary"], dict) or set(step["response_boundary"]) != {
            "event",
            "phase",
        }:
            raise RuntimeError(f"{label} has invalid settlement boundary evidence")
        for key in (
            "response_audio_deliverable",
            "assistant_text_unchanged",
            "function_text_unchanged",
            "function_cycle_effective",
        ):
            if not _is_bool(step[key]):
                raise RuntimeError(f"{label} has invalid settlement activity evidence")
        if not isinstance(step["assistant_delta"], str) or not isinstance(
            step["function_delta"], str
        ):
            raise RuntimeError(f"{label} has invalid settlement channel evidence")
        function_state = step["function_calling"]
        if not isinstance(function_state, dict) or set(function_state) != set(
            DIRECT_SEMANTIC_FUNCTION_STATE_KEYS
        ):
            raise RuntimeError(f"{label} has incomplete settlement function evidence")
        if any(
            not _is_bool(function_state[key])
            for key in (
                "active",
                "awaiting_response",
                "injecting_response",
                "background_active",
            )
        ) or not (
            _is_int(function_state["forced_tokens"]) and function_state["forced_tokens"] >= 0
        ):
            raise RuntimeError(f"{label} has invalid settlement function state")
        if not _validate_sotc_evidence(
            {key: function_state[key] for key in DIRECT_SEMANTIC_SOTC_KEYS}
        ):
            raise RuntimeError(f"{label} has invalid settlement SOTC evidence")
        if not isinstance(step["vllm_request_positions"], dict):
            raise RuntimeError(f"{label} has incomplete settlement model evidence")
        if not isinstance(step["violations"], list) or any(
            not isinstance(violation, str) for violation in step["violations"]
        ):
            raise RuntimeError(f"{label} has invalid settlement violations")
        boundary_state = step["response_boundary"]
        expected_violations = []
        if boundary_state["event"] is not None or boundary_state["phase"] not in {
            None,
            "idle",
        }:
            expected_violations.append("response_boundary_open")
        if step["agent_control"] != "pad":
            expected_violations.append("agent_output_effective")
        if step["response_audio_deliverable"]:
            expected_violations.append("response_audio_deliverable")
        if step["assistant_delta"] or not step["assistant_text_unchanged"]:
            expected_violations.append("assistant_text_mutated")
        if step["function_delta"] or not step["function_text_unchanged"]:
            expected_violations.append("function_text_mutated")
        if step["function_cycle_effective"]:
            expected_violations.append("function_cycle_effective")
        derived_function_effective = any(
            function_state[key]
            for key in (
                "active",
                "awaiting_response",
                "injecting_response",
                "forced_tokens",
                "background_active",
            )
        )
        if step["function_cycle_effective"] is not bool(derived_function_effective):
            raise RuntimeError(f"{label} has contradictory settlement function evidence")
        if step["violations"] != expected_violations:
            raise RuntimeError(f"{label} has contradictory settlement guard evidence")
    derived_fence = settlement["ending_blank_frames"] >= settlement["target_blank_frames"]
    derived_bound = settlement["model_steps"] <= settlement["max_model_steps"]
    derived_clean = all(not step["violations"] for step in steps)
    expected_ending = (
        steps[-1]["rnnt_blank_frames"] if steps else settlement["starting_blank_frames"]
    )
    if settlement["ending_blank_frames"] != expected_ending:
        raise RuntimeError(f"{label} has contradictory settlement blank counts")
    if settlement["fence_reached"] is not derived_fence:
        raise RuntimeError(f"{label} has an inconsistent settlement fence verdict")
    if settlement["within_bound"] is not derived_bound:
        raise RuntimeError(f"{label} has an inconsistent settlement bound verdict")
    if settlement["pre_eou_clean"] is not derived_clean:
        raise RuntimeError(f"{label} has an inconsistent settlement clean verdict")

    failure = case.get("settlement_failure")
    if failure is not None and (
        not isinstance(failure, dict)
        or set(failure) != {"code", "message"}
        or not isinstance(failure["code"], str)
        or not isinstance(failure["message"], str)
    ):
        raise RuntimeError(f"{label} has invalid settlement failure evidence")
    if (failure is None) is not (derived_fence and derived_bound and derived_clean):
        raise RuntimeError(f"{label} has a contradictory settlement outcome")
    if failure is not None:
        expected_code = (
            "input_turn_settle_timeout"
            if derived_clean and not derived_fence
            else (
                "pre_eou_function_activity"
                if steps and steps[-1]["function_cycle_effective"]
                else "pre_eou_response_activity"
            )
        )
        if failure["code"] != expected_code:
            raise RuntimeError(f"{label} has a contradictory settlement failure code")

    first_pcm = case.get("first_settlement_pcm")
    if first_pcm != (steps[0] if steps else None):
        raise RuntimeError(f"{label} has inconsistent first-settlement-PCM evidence")
    boundary = case.get("eou_boundary")
    if not isinstance(boundary, dict) or set(boundary) != set(DIRECT_SEMANTIC_EOU_BOUNDARY_KEYS):
        raise RuntimeError(f"{label} has incomplete EOU-boundary evidence")
    if not _is_bool(boundary["performed"]):
        raise RuntimeError(f"{label} has invalid EOU-boundary verdict")
    if boundary["performed"] is not (failure is None):
        raise RuntimeError(f"{label} has contradictory EOU-boundary evidence")
    if not _validate_sotc_evidence(boundary["sotc_before"]):
        raise RuntimeError(f"{label} has invalid pre-boundary SOTC evidence")
    if boundary["performed"]:
        for key in (
            "audio_frame_index_before",
            "audio_frame_index_after",
            "perception_frame_idx_before",
            "perception_frame_idx_after",
            "frame_index",
            "eartts_reset_count_after",
        ):
            if not _is_int(boundary[key]):
                raise RuntimeError(f"{label} has invalid EOU-boundary counters")
        if not isinstance(boundary["response_boundary"], dict) or not isinstance(
            boundary["vllm_request_positions"], dict
        ):
            raise RuntimeError(f"{label} has incomplete EOU-boundary model evidence")
        if not _validate_sotc_evidence(boundary["sotc_after"]) or not (
            _validate_sotc_evidence(boundary["sotc_final"])
        ):
            raise RuntimeError(f"{label} has invalid post-boundary SOTC evidence")
    else:
        for key in (
            "audio_frame_index_after",
            "perception_frame_idx_after",
            "frame_index",
            "eartts_reset_count_after",
            "agent_control",
            "response_boundary",
            "vllm_request_positions",
            "sotc_after",
            "sotc_final",
        ):
            if boundary[key] is not None:
                raise RuntimeError(f"{label} has contradictory skipped-boundary evidence")


def run_checked(
    command: list[str],
    *,
    timeout: float,
    log: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    if log is None:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, timeout=timeout)
        return
    with log.open("wb") as output:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def run_json_report_command(
    command: list[str], *, log: Path, report_path: Path, timeout: float
) -> tuple[dict[str, Any], int]:
    """Run a fail-closed gate whose exit code is the retained report verdict."""

    with log.open("wb") as output:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    if not report_path.is_file():
        raise RuntimeError(f"gate exited {completed.returncode} without {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"gate report is unreadable: {report_path}") from exc
    if not isinstance(report, dict) or type(report.get("passed")) is not bool:
        raise RuntimeError(f"gate report has no boolean verdict: {report_path}")
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"gate exited unexpectedly: {completed.returncode}")
    if (completed.returncode == 0) is not report["passed"]:
        raise RuntimeError("gate exit status contradicts its retained report")
    return report, completed.returncode


def run_diagnostic_report_command(
    command: list[str], *, log: Path, report_path: Path, timeout: float
) -> tuple[dict[str, Any], int]:
    """Run a deliberately red diagnostic whose exit means evidence completeness."""

    with log.open("wb") as output:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    if not report_path.is_file():
        raise RuntimeError(
            f"diagnostic exited {completed.returncode} without {report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"diagnostic report is unreadable: {report_path}") from exc
    if (
        not isinstance(report, dict)
        or type(report.get("diagnostic_complete")) is not bool
        or report.get("passed") is not False
        or report.get("promotion_eligible") is not False
    ):
        raise RuntimeError(f"diagnostic report contract is invalid: {report_path}")
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"diagnostic exited unexpectedly: {completed.returncode}")
    if (completed.returncode == 0) is not report["diagnostic_complete"]:
        raise RuntimeError("diagnostic exit status contradicts evidence completeness")
    return report, completed.returncode


def run_browser_gate(args: argparse.Namespace, environment: dict[str, str]) -> dict[str, Any]:
    """Run the browser gate with one bounded retry and retain every outcome."""

    command = [
        str(args.python),
        "-m",
        "pytest",
        "-q",
        "tests/e2e/test_browser_smallwebrtc_e2e.py::test_live_playground_typed_turn_and_audio",
    ]
    report_path = args.output / "browser-attempts.json"
    report: dict[str, Any] = {
        "passed": False,
        "max_attempts": 2,
        "policy": "bounded retry for documented upstream wrong-tool base rate",
        "attempts": [],
    }
    for attempt in range(1, report["max_attempts"] + 1):
        log_path = args.output / f"browser-attempt-{attempt}.log"
        started = time.monotonic()
        outcome: dict[str, Any] = {
            "attempt": attempt,
            "log": log_path.name,
            "passed": False,
        }
        try:
            with log_path.open("wb") as output:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=600,
                )
            outcome["returncode"] = completed.returncode
            outcome["passed"] = completed.returncode == 0
        except subprocess.TimeoutExpired:
            outcome["timed_out"] = True
        finally:
            outcome["duration_seconds"] = round(time.monotonic() - started, 3)
            if log_path.is_file():
                outcome["log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
            report["attempts"].append(outcome)
            report["passed"] = any(item["passed"] for item in report["attempts"])
            atomic_json(report_path, report)
        if outcome["passed"]:
            return report
        if attempt < report["max_attempts"]:
            wait_model_idle(args.model_port)

    raise RuntimeError(
        f"browser gate failed {report['max_attempts']} recorded attempts; see {report_path}"
    )


def up_command(
    args: argparse.Namespace, *, trace_fc_async_heartbeat: bool = False
) -> list[str]:
    command = [
        str(REPO_ROOT / "voicechat"),
        "--cache-root",
        str(args.cache_root),
        "--trace-root",
        str(args.trace_root),
        "up",
        "--model-port",
        str(args.model_port),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.pipecat_port),
    ]
    if trace_fc_async_heartbeat:
        command.append("--trace-fc-async-heartbeat")
    return command


def request_model_thread_dump(
    *,
    container_name: str,
    stack_log_path: Path,
    stack_log: Any,
    wait_seconds: float = 3.0,
) -> dict[str, Any]:
    """Request and verify an opt-in Python all-thread dump before teardown."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container_name):
        raise ValueError("configured model container name is not safe for docker kill")
    marker = b"\n=== FC ASYNC THREAD DUMP REQUESTED ===\n"
    stack_log.write(marker)
    stack_log.flush()
    marker_end = stack_log_path.stat().st_size
    command = ["docker", "kill", "--signal=USR1", container_name]
    result: dict[str, Any] = {
        "command": command,
        "returncode": None,
        "dump_observed": False,
        "marker_end_offset": marker_end,
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
        if completed.returncode != 0:
            return result
        deadline = time.monotonic() + max(0.0, min(wait_seconds, 5.0))
        while time.monotonic() < deadline:
            payload = stack_log_path.read_bytes()[marker_end:]
            if b"Current thread " in payload or b"Thread 0x" in payload:
                result["dump_observed"] = True
                result["observed_log_bytes"] = len(payload)
                break
            time.sleep(0.05)
    except (OSError, subprocess.SubprocessError) as exc:
        result["error_type"] = type(exc).__name__
    return result


def wait_for_playground(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"stack exited before Playground readiness: {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/client/", timeout=2):
                return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Playground readiness timed out: {last_error}")


def wait_model_idle(port: int, timeout: float = 30) -> dict[str, Any]:
    """Wait for the strict single-client model socket to become available."""

    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_health: dict[str, Any] | None = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                last_health = json.load(response)
            if last_health.get("status") == "ready" and not last_health.get("active_client"):
                return last_health
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(
        f"model did not become idle within {timeout}s: health={last_health}, error={last_error}"
    )


def asr_command(
    args: argparse.Namespace,
    run_dir: Path,
    report_name: str,
    *,
    max_silent_rate: float | None = None,
    output_dir: Path | None = None,
) -> tuple[list[str], Path]:
    output = run_dir.resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "none",
        "-e",
        f"VOICECHAT_ASR_EVALUATOR_IMAGE_ID={args.asr_evaluator_image_id}",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        (
            f"{output}:/qualification-input:ro"
            if output_dir is not None
            else f"{output}:/qualification"
        ),
    ]
    if output_dir is not None:
        command.extend(("-v", f"{output_dir.resolve()}:/qualification-output"))
    command.extend(
        [
        args.asr_evaluator_image_id,
        "python3",
        "-P",
        "/workspace/project/tools/qualification/transcribe_sustained.py",
        "--run-dir",
        "/qualification-input" if output_dir is not None else "/qualification",
        ]
    )
    if output_dir is not None:
        command.extend(("--output-dir", "/qualification-output"))
    if max_silent_rate is not None:
        command.extend(("--max-silent-rate", str(max_silent_rate)))
    return command, args.output / report_name


def tool_freshness_fixture_command(
    args: argparse.Namespace, *, runtime_image_id: str
) -> list[str]:
    fixture_dir = (args.output / "fixtures").resolve()
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "PYTHONPATH=/workspace/project/src",
        "-e",
        "HF_HOME=/models/huggingface",
        "-e",
        "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{(args.cache_root / 'huggingface').resolve()}:/models/huggingface:ro",
        "-v",
        f"{fixture_dir}:/qualification",
        runtime_image_id,
        "python3",
        "/workspace/project/tools/qualification/generate_tool_freshness_fixtures.py",
        "--output",
        "/qualification",
        "--runtime-image",
        args.runtime_image,
        "--runtime-image-id",
        runtime_image_id,
    ]


def tool_freshness_arm_command(
    args: argparse.Namespace,
    *,
    runtime_image_id: str,
    nonpromotable_reverse_order: bool = False,
) -> list[str]:
    command = [
        str(args.python),
        str(REPO_ROOT / "tools/qualification/tool_freshness_parity.py"),
        "--url",
        f"ws://127.0.0.1:{args.model_port}/v1/realtime",
        "--health-url",
        f"http://127.0.0.1:{args.model_port}/health",
        "--fixture-manifest",
        str(args.output / "fixtures/manifest.json"),
        "--runtime-image-id",
        runtime_image_id,
        "--output",
        str(args.output / "runtime"),
    ]
    if nonpromotable_reverse_order:
        command.append("--nonpromotable-reverse-order")
    return command


def tool_freshness_compare_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "tools/qualification/compare_tool_freshness_modalities.py"),
        "--typed-report",
        str(args.output / "asr/typed/report.json"),
        "--speech-report",
        str(args.output / "asr/speech/report.json"),
        "--runtime-report",
        str(args.output / "runtime/report.json"),
        "--policy",
        "parity_margin_v2",
        "--output",
        str(args.output / "comparison.json"),
    ]


def tool_freshness_reverse_analyzer_command(
    args: argparse.Namespace, *, preflight_only: bool = False
) -> list[str]:
    command = [
        str(args.python),
        str(REPO_ROOT / "tools/qualification/analyze_tool_freshness_reverse.py"),
        "--forward-comparison",
        str(args.reverse_forward_comparison),
        "--forward-fixture",
        str(args.reverse_forward_fixture),
    ]
    if preflight_only:
        command.extend(
            ("--preflight-only", "--output", str(args.output / "preflight.json"))
        )
    else:
        command.extend(
            (
                "--reverse-typed-report",
                str(args.output / "asr/typed/report.json"),
                "--reverse-speech-report",
                str(args.output / "asr/speech/report.json"),
                "--reverse-runtime-report",
                str(args.output / "runtime/report.json"),
                "--reverse-fixture",
                str(args.output / "fixtures/manifest.json"),
                "--output",
                str(args.output / "diagnostic.json"),
            )
        )
    return command


def gpu_container_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--network",
        "none",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{args.cache_root / 'artifacts' / 'parent'}:/models/voicechat:ro",
        "-v",
        (
            f"{args.cache_root / 'artifacts' / 'nano-skeleton'}:"
            "/models/NVIDIA-Nemotron-Nano-9B-v2:ro"
        ),
        "-v",
        f"{args.cache_root / 'artifacts' / 'release'}:/models/derived:ro",
    ]


def eartts_asr_command(args: argparse.Namespace, components: Path, name: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "none",
        "-e",
        f"VOICECHAT_ASR_EVALUATOR_IMAGE_ID={args.asr_evaluator_image_id}",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{components}:/qualification",
        args.asr_evaluator_image_id,
        "python3",
        "-P",
        "/workspace/project/tools/qualification/finalize_eartts_component_asr.py",
        "--output-dir",
        f"/qualification/{name}",
    ]


def direct_semantic_baseline_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "tools/qualification/typed_semantic_baseline_v3.py"),
        "--url",
        f"ws://127.0.0.1:{args.model_port}/v1/realtime",
        "--corpus",
        str(args.direct_semantic_corpus),
        "--output",
        str(args.output / "direct-text-semantics/pocket"),
        "--evidence-only",
    ]


def write_fp32_qualification_manifest(source: Path, destination: Path) -> dict[str, Any]:
    """Relocate only FP32 component paths for the read-only container mount."""
    source_bytes = source.read_bytes()
    manifest = json.loads(source_bytes)
    if manifest.get("kind") != "exact_public_vllm_extraction":
        raise RuntimeError("FP32 qualification requires an exact_public_vllm_extraction manifest")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != set(
        FP32_QUALIFICATION_COMPONENT_PATHS
    ):
        raise RuntimeError("FP32 qualification manifest has unexpected components")

    relocated = copy.deepcopy(manifest)
    relocation_components: dict[str, dict[str, str]] = {}
    for name, paths in FP32_QUALIFICATION_COMPONENT_PATHS.items():
        component = components.get(name)
        if not isinstance(component, dict) or component.get("path") != paths["source"]:
            raise RuntimeError(f"FP32 qualification manifest has unexpected {name} component path")
        relocated["components"][name]["path"] = paths["container"]
        relocation_components[name] = {
            "source_path": paths["source"],
            "container_path": paths["container"],
        }

    relocated["qualification_manifest_relocation"] = {
        "schema": 1,
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "mutation_policy": "component_paths_only",
        "components": relocation_components,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, relocated)
    if source.read_bytes() != source_bytes:
        raise RuntimeError("source FP32 manifest changed while preparing qualification")
    return relocated


def direct_semantic_probe_command(
    args: argparse.Namespace,
    *,
    precision: str,
    runtime_image_id: str,
    output_relative: Path | None = None,
    token_forms: tuple[str, ...] = DIRECT_SEMANTIC_TOKEN_FORMS,
    function_logit_trace_top_k: int = 0,
) -> list[str]:
    if precision not in {"fp32", "w8"}:
        raise ValueError(f"unsupported direct semantic precision: {precision}")
    if not token_forms or set(token_forms) - set(DIRECT_SEMANTIC_TOKEN_FORMS):
        raise ValueError("unsupported direct semantic token form")
    if not 0 <= function_logit_trace_top_k <= 20:
        raise ValueError("function-logit trace top-k must be from 0 through 20")
    relative = output_relative or Path(f"direct-text-semantics/{precision}")
    output = (args.output / relative).resolve()
    environment = load_config()["runtime"]["environment"] | {
        "HF_HOME": "/models/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/models/huggingface/hub",
        "HF_MODULES_CACHE": "/tmp/voicechat-hf-modules",
        "VOICECHAT_DIRECT_POSITION_PROBE_SKIP_WARMUPS": "1",
        "VOICECHAT_DIRECT_SEMANTIC_CORPUS": "/qualification-input/corpus.json",
        "VOICECHAT_DIRECT_SEMANTIC_FORMS": ",".join(token_forms),
        "VOICECHAT_DIRECT_SEMANTIC_MAX_FRAMES": "240",
        "VOICECHAT_DIRECT_SEMANTIC_PRECISION": precision,
        "VOICECHAT_DIRECT_SEMANTIC_PROBE_DIR": "/qualification",
        "VOICECHAT_NANO_PAD_PAIR": "0",
        "VOICECHAT_RUNTIME_IMAGE": args.runtime_image,
        "VOICECHAT_RUNTIME_IMAGE_ID": runtime_image_id,
        "VOICECHAT_WEB_REALTIME_WARMUP": "0",
    }
    if function_logit_trace_top_k:
        environment["S2S_FUNCTION_LOGIT_TRACE_TOPK"] = str(function_logit_trace_top_k)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"voicechat-direct-semantic-{precision}",
        "--init",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--network",
        "none",
        "--security-opt",
        "label=disable",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{args.cache_root / 'artifacts' / 'parent'}:/models/voicechat:ro",
        "-v",
        (
            f"{args.cache_root / 'artifacts' / 'nano-skeleton'}:"
            "/models/NVIDIA-Nemotron-Nano-9B-v2:ro"
        ),
        "-v",
        f"{args.cache_root / 'huggingface'}:/models/huggingface:ro",
        "-v",
        f"{output}:/qualification",
        "-v",
        f"{args.direct_semantic_corpus.resolve()}:/qualification-input/corpus.json:ro",
    ]
    diagnostic_speech_root = getattr(args, "diagnostic_speech_root", None)
    if diagnostic_speech_root is not None:
        command.extend(
            [
                "-v",
                f"{Path(diagnostic_speech_root).expanduser().resolve()}:/opt/Speech:ro",
                "-e",
                "GIT_CONFIG_COUNT=1",
                "-e",
                "GIT_CONFIG_KEY_0=safe.directory",
                "-e",
                "GIT_CONFIG_VALUE_0=/opt/Speech",
                "-e",
                "PYTHONPATH=/workspace/project/src:/opt/Speech",
            ]
        )
    if precision == "fp32":
        derived_root = args.fp32_derived_root.expanduser().resolve()
        relocated_manifest = (
            args.output / "direct-text-semantics/fp32-relocated-manifest.json"
        ).resolve()
        command.extend(
            [
                "-v",
                f"{derived_root}:/derived:ro",
                "-v",
                f"{relocated_manifest}:/qualification-input/fp32-manifest.json:ro",
            ]
        )
        manifest = "/qualification-input/fp32-manifest.json"
        nano = "/derived/nano-vllm-fp32"
        eartts = "/derived/eartts-vllm-fp32"
    else:
        command.extend(
            [
                "-v",
                f"{args.cache_root / 'artifacts' / 'release'}:/models/derived:ro",
            ]
        )
        manifest = "/models/derived/manifests/release.json"
        nano = "/models/derived/nano"
        eartts = "/models/derived/eartts"
    for name, value in sorted(environment.items()):
        command.extend(["-e", f"{name}={value}"])
    command.extend(
        [
            args.runtime_image,
            "python3",
            "-m",
            "nemotron_voicechat_runtime.server",
            "--speech-root",
            "/opt/Speech",
            "--checkpoint-root",
            "/models/voicechat",
            "--hf-skeleton",
            "/models/NVIDIA-Nemotron-Nano-9B-v2",
            "--vllm-manifest",
            manifest,
            "--nano-vllm-path",
            nano,
            "--eartts-vllm-path",
            eartts,
            "--speaker-name",
            "Aria",
        ]
    )
    return command


def validate_direct_semantic_probe_report(
    payload: dict[str, Any],
    *,
    precision: str,
    runtime_image_id: str,
    corpus_path: Path,
    expected_token_forms: tuple[str, ...] = DIRECT_SEMANTIC_TOKEN_FORMS,
    expected_trace_top_k: int = 0,
) -> None:
    """Reject partial/crashed probe artifacts without requiring semantic green."""

    if not 0 <= expected_trace_top_k <= 20:
        raise ValueError("expected function-logit trace top-k must be from 0 through 20")
    expected_schema = 3 if expected_trace_top_k else 2
    if (
        payload.get("schema") != expected_schema
        or payload.get("kind") != "direct_text_semantic_probe"
    ):
        raise RuntimeError(f"{precision} direct semantic report has an invalid identity")
    trace_config = payload.get("function_logit_trace")
    if not expected_trace_top_k:
        if trace_config is not None:
            raise RuntimeError(f"{precision} unexpectedly contains function-logit tracing")
        watched_token_ids = None
        post_bos_frames = 0
    else:
        if not isinstance(trace_config, dict) or set(trace_config) != {
            "top_k",
            "post_bos_frames",
            "phases",
            "watched_token_ids",
        }:
            raise RuntimeError(f"{precision} has incomplete function-logit trace config")
        if trace_config["top_k"] != expected_trace_top_k:
            raise RuntimeError(f"{precision} function-logit top-k mismatch")
        post_bos_frames = trace_config["post_bos_frames"]
        if not _is_int(post_bos_frames) or not 1 <= post_bos_frames <= 32:
            raise RuntimeError(f"{precision} has an invalid post-BOS trace bound")
        if trace_config["phases"] != list(DIRECT_SEMANTIC_FUNCTION_TRACE_PHASES):
            raise RuntimeError(f"{precision} has invalid function-logit trace phases")
        watched_token_ids = trace_config["watched_token_ids"]
        if (
            not isinstance(watched_token_ids, dict)
            or set(watched_token_ids) != set(DIRECT_SEMANTIC_FUNCTION_TRACE_WATCHED)
            or any(not _is_int(value) or value < 0 for value in watched_token_ids.values())
            or len(set(watched_token_ids.values())) != len(watched_token_ids)
        ):
            raise RuntimeError(f"{precision} has invalid watched function token IDs")
    if payload.get("error") is not None:
        raise RuntimeError(f"{precision} direct semantic probe reported an error")
    if payload.get("precision") != precision:
        raise RuntimeError(f"{precision} direct semantic report precision mismatch")
    if not _is_bool(payload.get("passed")):
        raise RuntimeError(f"{precision} direct semantic report has no boolean verdict")
    if not _is_bool(payload.get("structural_passed")):
        raise RuntimeError(f"{precision} direct semantic report has no structural verdict")
    if payload.get("corpus_sha256") != corpus_sha256(corpus_path):
        raise RuntimeError(f"{precision} direct semantic report corpus mismatch")
    reported_image_id = (payload.get("runtime_provenance") or {}).get("runtime_image_id")
    if reported_image_id != runtime_image_id:
        raise RuntimeError(
            f"{precision} report image ID mismatch: {reported_image_id!r} != {runtime_image_id!r}"
        )

    corpus = load_semantic_corpus(corpus_path)
    expected_case_ids = [case["id"] for case in corpus["cases"]]
    if not expected_token_forms or set(expected_token_forms) - set(DIRECT_SEMANTIC_TOKEN_FORMS):
        raise ValueError("expected direct semantic token forms are unsupported")
    if payload.get("token_forms") != list(expected_token_forms):
        raise RuntimeError(f"{precision} direct semantic token forms are incomplete")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or [lane.get("token_form") for lane in lanes] != list(
        expected_token_forms
    ):
        raise RuntimeError(f"{precision} direct semantic lanes are incomplete")

    lane_structural: list[bool] = []
    for lane in lanes:
        cases = lane.get("cases")
        if not isinstance(cases, list):
            raise RuntimeError(f"{precision}/{lane.get('token_form')} has no case list")
        case_ids = [case.get("case_id") for case in cases]
        if lane.get("case_count") != len(expected_case_ids) or case_ids != expected_case_ids:
            raise RuntimeError(f"{precision}/{lane.get('token_form')} case order/set is incomplete")
        if not _is_bool(lane.get("structural_passed")):
            raise RuntimeError(f"{precision}/{lane.get('token_form')} has no structural verdict")
        semantic_passes = 0
        case_structural: list[bool] = []
        for case in cases:
            semantic = case.get("semantic")
            structural = case.get("structural_passed")
            checks = case.get("structural_checks")
            if not isinstance(semantic, dict) or not _is_bool(semantic.get("passed")):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has no semantic verdict"
                )
            if not _is_bool(structural) or not isinstance(checks, dict):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has incomplete structural evidence"
                )
            if set(checks) != set(DIRECT_SEMANTIC_STRUCTURAL_CHECKS) or any(
                not _is_bool(checks.get(check)) for check in DIRECT_SEMANTIC_STRUCTURAL_CHECKS
            ):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has incomplete structural checks"
                )
            if structural is not all(checks.values()):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has an inconsistent structural verdict"
                )
            token_ids = case.get("token_ids")
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or any(not isinstance(token_id, int) for token_id in token_ids)
            ):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has no direct token evidence"
                )
            if not isinstance(case.get("direct_positions"), list):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has no direct-position evidence"
                )
            label = f"{precision}/{lane.get('token_form')}/{case.get('case_id')}"
            _validate_direct_settlement_evidence(case, label)
            if expected_trace_top_k:
                assert watched_token_ids is not None
                _validate_function_logit_trace(
                    case,
                    label,
                    top_k=expected_trace_top_k,
                    post_bos_frames=post_bos_frames,
                    watched_token_ids=watched_token_ids,
                )
            elif "function_logit_trace" in case:
                raise RuntimeError(f"{label} unexpectedly contains function-logit tracing")
            direct_state = case.get("direct_state")
            if (
                not isinstance(direct_state, dict)
                or set(direct_state)
                != {
                    "perception_frame_idx_before",
                    "perception_frame_idx_after",
                    "audio_frame_index_after",
                    "eartts_reset_count_before",
                    "eartts_reset_count_after",
                }
                or any(not _is_int(value) for value in direct_state.values())
            ):
                raise RuntimeError(f"{label} has invalid direct-state evidence")
            settlement = case["settlement"]
            settlement_steps = settlement["steps"]
            first_settlement = settlement_steps[0] if settlement_steps else None
            boundary = case["eou_boundary"]
            derived_checks = {
                "perception_held_during_direct": direct_state["perception_frame_idx_after"]
                == direct_state["perception_frame_idx_before"],
                "audio_clock_held_during_direct": direct_state["audio_frame_index_after"] == 0,
                "eartts_held_during_direct": direct_state["eartts_reset_count_after"]
                == direct_state["eartts_reset_count_before"],
                "first_settlement_pcm_initialized_perception": bool(
                    first_settlement
                    and first_settlement["audio_frame_index_after"]
                    == direct_state["audio_frame_index_after"] + 1
                    and first_settlement["perception_frame_idx_after"]
                    == direct_state["perception_frame_idx_after"] + 1
                ),
                "settlement_reached_blank_fence": settlement["fence_reached"],
                "settlement_within_bound": settlement["within_bound"],
                "settlement_pre_eou_clean": bool(
                    settlement["pre_eou_clean"] and case["settlement_failure"] is None
                ),
                "eartts_held_during_settlement": all(
                    step["eartts_reset_count_after"] == direct_state["eartts_reset_count_after"]
                    for step in settlement_steps
                )
                and settlement["eartts_reset_count_starting"]
                == direct_state["eartts_reset_count_after"],
                "eou_boundary_position_exactly_once": bool(
                    boundary["performed"]
                    and boundary["audio_frame_index_after"]
                    == boundary["audio_frame_index_before"] + 1
                    and boundary["perception_frame_idx_after"]
                    == boundary["perception_frame_idx_before"] + 1
                ),
                "eartts_bos_reset_exactly_once": bool(
                    boundary["performed"]
                    and boundary["eartts_reset_count_after"]
                    == direct_state["eartts_reset_count_after"] + 1
                ),
            }
            if any(checks[key] is not value for key, value in derived_checks.items()):
                raise RuntimeError(f"{label} has contradictory settlement structural checks")
            if not isinstance(case.get("assistant_text"), str) or not isinstance(
                case.get("function_text"), str
            ):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has no model-channel evidence"
                )
            audio_samples = case.get("audio_samples")
            if (
                not isinstance(audio_samples, int)
                or isinstance(audio_samples, bool)
                or audio_samples < 0
            ):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has no response-audio evidence"
                )
            if not isinstance(case.get("tool_calls"), list):
                raise RuntimeError(
                    f"{precision}/{lane.get('token_form')}/{case.get('case_id')} "
                    "has no tool-call evidence"
                )
            semantic_passes += int(semantic["passed"])
            case_structural.append(structural)
        if lane.get("semantic_pass_count") != semantic_passes:
            raise RuntimeError(
                f"{precision}/{lane.get('token_form')} semantic count is inconsistent"
            )
        derived_lane_structural = all(case_structural)
        if lane["structural_passed"] is not derived_lane_structural:
            raise RuntimeError(
                f"{precision}/{lane.get('token_form')} structural verdict is inconsistent"
            )
        lane_structural.append(derived_lane_structural)

    derived_structural = all(lane_structural)
    if payload["structural_passed"] is not derived_structural:
        raise RuntimeError(f"{precision} top-level structural verdict is inconsistent")
    if payload["passed"] is not derived_structural:
        raise RuntimeError(f"{precision} top-level probe verdict is inconsistent")


FUNCTION_LOGIT_DIAGNOSTIC_CASE_ID = "c014"
EARTTS_FRAME_SAMPLES = 1_764


def _normalize_function_logit_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip trace-only and process-volatile evidence at every report depth."""

    normalized = copy.deepcopy(payload)
    schema = normalized.get("schema")
    if schema not in {2, 3}:
        raise RuntimeError("function-logit neutrality comparison requires schema 2 or 3")
    normalized["schema"] = 2

    trace_keys = 0

    def visit(value: Any) -> None:
        nonlocal trace_keys
        if isinstance(value, dict):
            if "function_logit_trace" in value:
                trace_keys += 1
                value.pop("function_logit_trace")
            for key, child in list(value.items()):
                if key == "audio_path" and isinstance(child, str):
                    value[key] = Path(child).name
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    if schema == 3 and trace_keys < 2:
        raise RuntimeError("traced report has no nested function-logit evidence")
    if schema == 2 and trace_keys:
        raise RuntimeError("untraced report unexpectedly contains function-logit evidence")

    checkpoint = normalized.get("runtime_provenance", {}).get("checkpoint", {})
    fixed = checkpoint.get("fixed_stage_optimizations", {}) if isinstance(checkpoint, dict) else {}
    cpu_codec = fixed.get("cpu_codec", {}) if isinstance(fixed, dict) else {}
    if isinstance(cpu_codec, dict):
        cpu_codec.pop("warmup_worker_ms", None)
        worker = cpu_codec.get("worker")
        if isinstance(worker, dict):
            worker.pop("pid", None)
    return normalized


def _case_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    lanes = payload.get("lanes")
    if not isinstance(lanes, list):
        raise RuntimeError("function-logit neutrality report has no lanes")
    for lane in lanes:
        token_form = lane.get("token_form")
        lane_cases = lane.get("cases")
        if not isinstance(token_form, str) or not isinstance(lane_cases, list):
            raise RuntimeError("function-logit neutrality lane is malformed")
        for case in lane_cases:
            case_id = case.get("case_id")
            key = (token_form, case_id)
            if not isinstance(case_id, str) or key in cases:
                raise RuntimeError("function-logit neutrality cases are ambiguous")
            cases[key] = case
    return cases


def _bounded_field_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "bytes": len(encoded)}


def _classify_trace_value(
    control_a: Any,
    traced: Any,
    control_b: Any,
    *,
    path: tuple[str | int, ...],
    evidence: dict[str, Any],
) -> None:
    if all(isinstance(value, dict) for value in (control_a, traced, control_b)):
        key_sets = [set(value) for value in (control_a, traced, control_b)]
        if key_sets[0] == key_sets[1] == key_sets[2]:
            for key in sorted(key_sets[0]):
                _classify_trace_value(
                    control_a[key],
                    traced[key],
                    control_b[key],
                    path=(*path, key),
                    evidence=evidence,
                )
            return
    if all(isinstance(value, list) for value in (control_a, traced, control_b)):
        lengths = {len(control_a), len(traced), len(control_b)}
        if len(lengths) == 1:
            for index, values in enumerate(zip(control_a, traced, control_b, strict=True)):
                _classify_trace_value(
                    *values,
                    path=(*path, index),
                    evidence=evidence,
                )
            return

    path_text = "/".join(str(part) for part in path)
    if control_a == control_b:
        evidence["stable_field_count"] += 1
        if traced != control_a:
            evidence["violations"].append(
                {
                    "path": path_text,
                    "reason": "trace_changed_control_stable_field",
                    "control": _bounded_field_value(control_a),
                    "traced": _bounded_field_value(traced),
                }
            )
        return

    evidence["control_nondeterministic_field_count"] += 1
    if (
        isinstance(control_a, (int, float))
        and not isinstance(control_a, bool)
        and isinstance(control_b, (int, float))
        and not isinstance(control_b, bool)
        and isinstance(traced, (int, float))
        and not isinstance(traced, bool)
    ):
        within_null = min(control_a, control_b) <= traced <= max(control_a, control_b)
        rule = "numeric_range"
    else:
        within_null = traced == control_a or traced == control_b
        rule = "observed_set"
    classification = {
        "path": path_text,
        "rule": rule,
        "control_a": _bounded_field_value(control_a),
        "control_b": _bounded_field_value(control_b),
        "traced": _bounded_field_value(traced),
        "within_observed_null": within_null,
    }
    evidence["control_nondeterministic_fields"].append(classification)
    if not within_null:
        evidence["violations"].append(
            {**classification, "reason": "trace_outside_observed_control_null"}
        )


def validate_function_logit_trace_neutrality(
    control_a: dict[str, Any], traced: dict[str, Any], control_b: dict[str, Any]
) -> dict[str, Any]:
    """Classify traced evidence against a contemporaneous untraced ABA null."""

    if control_a.get("schema") != 2 or traced.get("schema") != 3 or control_b.get("schema") != 2:
        raise RuntimeError("function-logit neutrality comparison requires schema 2/3/2 ABA")
    normalized = [
        _normalize_function_logit_report(payload)
        for payload in (control_a, traced, control_b)
    ]
    case_maps = [_case_map(payload) for payload in normalized]
    if not (set(case_maps[0]) == set(case_maps[1]) == set(case_maps[2])):
        raise RuntimeError("function-logit neutrality reports cover different cases")

    audio_evidence = []
    for key in sorted(case_maps[0]):
        samples = [case_maps[index][key].pop("audio_samples", None) for index in range(3)]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in samples
        ):
            raise RuntimeError("function-logit neutrality has invalid response-audio evidence")
        if any(value % EARTTS_FRAME_SAMPLES for value in samples):
            raise RuntimeError("function-logit neutrality audio is not quantized to EarTTS frames")
        control_delta = abs(samples[2] - samples[0]) // EARTTS_FRAME_SAMPLES
        traced_delta = abs(samples[1] - samples[0]) // EARTTS_FRAME_SAMPLES
        audio_evidence.append(
            {
                "token_form": key[0],
                "case_id": key[1],
                "control_a_samples": samples[0],
                "traced_samples": samples[1],
                "control_b_samples": samples[2],
                "control_delta_frames": control_delta,
                "traced_delta_frames": traced_delta,
                "within_observed_null": (
                    min(samples[0], samples[2]) <= samples[1] <= max(samples[0], samples[2])
                    and traced_delta <= control_delta
                ),
            }
        )

    evidence: dict[str, Any] = {
        "schema": 1,
        "kind": "function_logit_trace_neutrality",
        "method": "contemporaneous_off_on_off_observed_null",
        "stable_field_count": 0,
        "control_nondeterministic_field_count": 0,
        "control_nondeterministic_fields": [],
        "audio_tail": audio_evidence,
        "violations": [],
    }
    _classify_trace_value(
        normalized[0], normalized[1], normalized[2], path=(), evidence=evidence
    )
    evidence["violations"].extend(
        {
            **entry,
            "path": f"lanes/{entry['token_form']}/{entry['case_id']}/audio_samples",
            "reason": "trace_audio_tail_outside_observed_control_null",
        }
        for entry in audio_evidence
        if not entry["within_observed_null"]
    )

    original_cases = [_case_map(payload) for payload in (control_a, traced, control_b)]
    diagnostic_key = ("text", FUNCTION_LOGIT_DIAGNOSTIC_CASE_ID)
    if any(diagnostic_key not in cases for cases in original_cases):
        raise RuntimeError("function-logit neutrality report has no c014 text case")
    diagnostic_cases = [cases[diagnostic_key] for cases in original_cases]
    discrete_fields = ("assistant_text", "function_text", "tool_calls")
    discrete_equal = all(
        diagnostic_cases[0].get(field)
        == diagnostic_cases[1].get(field)
        == diagnostic_cases[2].get(field)
        for field in discrete_fields
    )
    trace_entries = diagnostic_cases[1].get("function_logit_trace")
    if not isinstance(trace_entries, list):
        raise RuntimeError("traced c014 has no function-logit entries")
    settlement = [entry for entry in trace_entries if entry.get("phase") == "settlement"]
    boundary = [entry for entry in trace_entries if entry.get("phase") == "eou_boundary"]
    pad_id = traced["function_logit_trace"]["watched_token_ids"]["pad"]
    rank_two_pad_argmax = bool(settlement and len(boundary) == 1) and all(
        entry.get("sotc_rank") == 2
        and entry.get("head_argmax_id") == pad_id
        and entry["watched"]["pad"]["logit"] > entry["watched"]["sotc"]["logit"]
        for entry in [*settlement, *boundary]
    )
    evidence["c014"] = {
        "discrete_fields": list(discrete_fields),
        "discrete_evidence_equal": discrete_equal,
        "settlement_trace_count": len(settlement),
        "eou_boundary_trace_count": len(boundary),
        "settlement_and_eou_sotc_rank_two_pad_argmax": rank_two_pad_argmax,
    }
    if not discrete_equal:
        evidence["violations"].append(
            {"path": "c014", "reason": "c014_discrete_evidence_is_not_control_stable"}
        )
    if not rank_two_pad_argmax:
        evidence["violations"].append(
            {"path": "c014", "reason": "c014_trace_does_not_support_rank_two_finding"}
        )
    evidence["passed"] = not evidence["violations"]
    return evidence


def run_direct_semantic_probe_stage(
    command: list[str],
    *,
    log: Path,
    report_path: Path,
    precision: str,
    runtime_image_id: str,
    corpus_path: Path,
    timeout: float,
    expected_token_forms: tuple[str, ...] = DIRECT_SEMANTIC_TOKEN_FORMS,
    expected_trace_top_k: int = 0,
) -> tuple[dict[str, Any], int]:
    """Run one precision lane and preserve a complete red report as evidence."""

    with log.open("wb") as output:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    if not report_path.is_file():
        raise RuntimeError(
            f"{precision} direct semantic container exited {completed.returncode} without a report"
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{precision} direct semantic report is unreadable") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{precision} direct semantic report must be an object")
    validate_direct_semantic_probe_report(
        payload,
        precision=precision,
        runtime_image_id=runtime_image_id,
        corpus_path=corpus_path,
        expected_token_forms=expected_token_forms,
        expected_trace_top_k=expected_trace_top_k,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"{precision} direct semantic container exited unexpectedly: {completed.returncode}"
        )
    if completed.returncode == 0 and payload["passed"] is not True:
        raise RuntimeError(f"{precision} direct semantic container exited zero with a red report")
    if completed.returncode == 1 and payload["passed"] is not False:
        raise RuntimeError(f"{precision} direct semantic container failed despite a green report")
    return payload, completed.returncode


def validate_direct_semantic_comparison_report(payload: dict[str, Any]) -> None:
    """Distinguish an honest red comparison from evaluator failure."""

    if payload.get("schema") != 1 or payload.get("kind") != "direct_text_semantic_comparison":
        raise RuntimeError("direct semantic comparison has an invalid identity")
    if payload.get("error") is not None or not _is_bool(payload.get("passed")):
        raise RuntimeError("direct semantic comparison is incomplete")
    expected_lanes = {
        f"{precision}/{token_form}"
        for precision in ("fp32", "w8")
        for token_form in DIRECT_SEMANTIC_TOKEN_FORMS
    }
    lanes = payload.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != expected_lanes:
        raise RuntimeError("direct semantic comparison lanes are incomplete")
    lane_gate_fields = (
        "structural_passed",
        "hidden_transaction_rederived_passed",
        "direct_intent_predicate_passed",
        "tool_name_and_arguments_passed",
        "safety_critical_passed",
    )
    for lane_name, lane in lanes.items():
        if not isinstance(lane, dict) or any(
            not _is_bool(lane.get(field)) for field in (*lane_gate_fields, "passed")
        ):
            raise RuntimeError(
                f"direct semantic comparison lane {lane_name} verdicts are incomplete"
            )
        derived_lane_passed = all(lane[field] for field in lane_gate_fields)
        if lane["passed"] is not derived_lane_passed:
            raise RuntimeError(
                f"direct semantic comparison lane {lane_name} verdict is inconsistent"
            )
    token_forms = payload.get("token_forms")
    if not isinstance(token_forms, dict) or set(token_forms) != set(DIRECT_SEMANTIC_TOKEN_FORMS):
        raise RuntimeError("direct semantic comparison token forms are incomplete")
    for token_form, result in token_forms.items():
        if not isinstance(result, dict) or any(
            not _is_bool(result.get(field))
            for field in (
                "fp32_lane_passed",
                "w8_lane_passed",
                "w8_regression_gate_passed",
                "qualified",
            )
        ):
            raise RuntimeError(
                f"direct semantic comparison {token_form} qualification is incomplete"
            )
        if result["fp32_lane_passed"] is not lanes[f"fp32/{token_form}"]["passed"]:
            raise RuntimeError(
                f"direct semantic comparison {token_form} FP32 verdict is inconsistent"
            )
        if result["w8_lane_passed"] is not lanes[f"w8/{token_form}"]["passed"]:
            raise RuntimeError(
                f"direct semantic comparison {token_form} W8 verdict is inconsistent"
            )
        derived_qualified = bool(
            result["fp32_lane_passed"]
            and result["w8_lane_passed"]
            and result["w8_regression_gate_passed"]
        )
        if result["qualified"] is not derived_qualified:
            raise RuntimeError(
                f"direct semantic comparison {token_form} qualification is inconsistent"
            )
    for field in (
        "corpus_identity_passed",
        "runtime_provenance_passed",
        "source_provenance_consistent",
        "structural_all_forms_passed",
    ):
        if not _is_bool(payload.get(field)):
            raise RuntimeError(f"direct semantic comparison has no {field} verdict")
    pocket = payload.get("pocket")
    if not isinstance(pocket, dict) or not _is_bool(pocket.get("evidence_complete")):
        raise RuntimeError("direct semantic comparison has no Pocket evidence verdict")
    derived_structural = all(lane["structural_passed"] for lane in lanes.values())
    if payload["structural_all_forms_passed"] is not derived_structural:
        raise RuntimeError("direct semantic comparison structural verdict is inconsistent")
    qualified = [
        token_form
        for token_form in DIRECT_SEMANTIC_TOKEN_FORMS
        if token_forms[token_form]["qualified"]
    ]
    if payload.get("qualified_token_forms") != qualified:
        raise RuntimeError("direct semantic comparison qualified forms are inconsistent")
    selected = qualified[0] if qualified else None
    if payload.get("selected_token_form") != selected:
        raise RuntimeError("direct semantic comparison selected form is inconsistent")
    derived_passed = bool(
        payload["corpus_identity_passed"]
        and payload["runtime_provenance_passed"]
        and payload["source_provenance_consistent"]
        and pocket["evidence_complete"]
        and payload["structural_all_forms_passed"]
        and selected is not None
    )
    if payload["passed"] is not derived_passed:
        raise RuntimeError("direct semantic comparison verdict is inconsistent")


def run_direct_semantic_comparison_stage(
    command: list[str],
    *,
    log: Path,
    report_path: Path,
    timeout: float,
) -> tuple[dict[str, Any], int]:
    """Run the evaluator while retaining an honest red comparison artifact."""

    with log.open("wb") as output:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    if not report_path.is_file():
        raise RuntimeError(
            f"direct semantic comparison exited {completed.returncode} without a report"
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("direct semantic comparison report is unreadable") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("direct semantic comparison report must be an object")
    validate_direct_semantic_comparison_report(payload)
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"direct semantic comparison exited unexpectedly: {completed.returncode}"
        )
    if completed.returncode == 0 and payload["passed"] is not True:
        raise RuntimeError("direct semantic comparison exited zero with a red report")
    if completed.returncode == 1 and payload["passed"] is not False:
        raise RuntimeError("direct semantic comparison failed despite a green report")
    return payload, completed.returncode


def runtime_image_id(image: str) -> str:
    value = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not valid_runtime_image_id(value):
        raise RuntimeError(f"runtime image has no immutable Docker ID: {value!r}")
    return value


def direct_semantic_image_preflight_command(image_ref: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python3",
        image_ref,
        "-c",
        (
            "import nemotron_voicechat_runtime.direct_semantic_probe; "
            "import nemotron_voicechat_runtime.semantic_corpus"
        ),
    ]


def run_direct_semantic_gpu_ladder(args: argparse.Namespace, *, image_id: str) -> dict[str, Any]:
    root = args.output / "direct-text-semantics"
    state_path = root / "ladder.json"
    state: dict[str, Any] = {
        "schema": 1,
        "kind": "direct_text_semantic_ladder",
        "runtime_image": args.runtime_image,
        "runtime_image_id": image_id,
        "corpus": str(args.direct_semantic_corpus.resolve()),
        "corpus_sha256": corpus_sha256(args.direct_semantic_corpus),
        "stages": [],
        "passed": False,
    }
    pocket_report = root / "pocket/report.json"
    pocket_payload = json.loads(pocket_report.read_text(encoding="utf-8"))
    pocket_image_id = (pocket_payload.get("runtime_provenance") or {}).get("runtime_image_id")
    if pocket_image_id != image_id:
        raise RuntimeError(f"Pocket report image ID mismatch: {pocket_image_id!r} != {image_id!r}")
    state["stages"].append(
        {
            "name": "pocket",
            "status": "completed",
            "barrier": "model_idle",
            "report": str(pocket_report),
            "report_sha256": hashlib.sha256(pocket_report.read_bytes()).hexdigest(),
            "reported_runtime_image_id": pocket_image_id,
        }
    )
    atomic_json(state_path, state)
    for precision in ("fp32", "w8"):
        output = root / precision
        output.mkdir()
        if precision == "fp32":
            write_fp32_qualification_manifest(
                args.fp32_derived_root / "manifest.json",
                root / "fp32-relocated-manifest.json",
            )
        command = direct_semantic_probe_command(
            args, precision=precision, runtime_image_id=image_id
        )
        stage = {
            "name": precision,
            "status": "running",
            "barrier": "previous_model_container_exited",
            "command": command,
        }
        state["stages"].append(stage)
        atomic_json(state_path, state)
        report_path = output / "report.json"
        stage_payload, container_returncode = run_direct_semantic_probe_stage(
            command,
            log=root / f"{precision}.log",
            report_path=report_path,
            precision=precision,
            runtime_image_id=image_id,
            corpus_path=args.direct_semantic_corpus,
            timeout=7200,
        )
        reported_image_id = (stage_payload.get("runtime_provenance") or {}).get("runtime_image_id")
        if reported_image_id != image_id:
            raise RuntimeError(
                f"{precision} report image ID mismatch: {reported_image_id!r} != {image_id!r}"
            )
        stage.update(
            {
                "status": "completed",
                "container_returncode": container_returncode,
                "probe_passed": stage_payload["passed"],
                "structural_passed": stage_payload["structural_passed"],
                "report": str(report_path),
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "reported_runtime_image_id": reported_image_id,
            }
        )
        atomic_json(state_path, state)
    comparison = root / "comparison.json"
    compare_command = [
        str(args.python),
        str(REPO_ROOT / "tools/qualification/evaluate_direct_text_semantics.py"),
        "--corpus",
        str(args.direct_semantic_corpus),
        "--pocket",
        str(root / "pocket/report.json"),
        "--fp32",
        str(root / "fp32/report.json"),
        "--w8",
        str(root / "w8/report.json"),
        "--output",
        str(comparison),
    ]
    state["stages"].append(
        {
            "name": "compare",
            "status": "running",
            "barrier": "w8_model_container_exited",
            "command": compare_command,
        }
    )
    atomic_json(state_path, state)
    report, comparison_returncode = run_direct_semantic_comparison_stage(
        compare_command,
        log=root / "comparison.log",
        report_path=comparison,
        timeout=300,
    )
    state["stages"][-1].update(
        {
            "status": "completed",
            "evaluator_returncode": comparison_returncode,
            "comparison_passed": report["passed"],
            "report": str(comparison),
            "report_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
        }
    )
    state["passed"] = report.get("passed") is True
    atomic_json(state_path, state)
    return report


def run_direct_semantics_only(args: argparse.Namespace) -> dict[str, Any]:
    if args.fp32_derived_root is None:
        raise RuntimeError("--direct-text-semantics-only requires --fp32-derived-root")
    config = load_config()
    configured_image = config["image"]["runtime"]
    if args.runtime_image != configured_image:
        raise RuntimeError(
            "direct semantic Pocket baseline must use the configured runtime image: "
            f"{args.runtime_image} != {configured_image}"
        )
    args.fp32_derived_root = args.fp32_derived_root.expanduser().resolve()
    args.direct_semantic_corpus = args.direct_semantic_corpus.expanduser().resolve()
    required_fp32 = (
        args.fp32_derived_root / "manifest.json",
        args.fp32_derived_root / "nano-vllm-fp32/config.json",
        args.fp32_derived_root / "eartts-vllm-fp32/config.json",
    )
    if missing := [str(path) for path in required_fp32 if not path.is_file()]:
        raise RuntimeError(f"FP32 direct-semantic artifacts are incomplete: {missing}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "direct-text-semantics").mkdir()
    image_id = runtime_image_id(args.runtime_image)
    run_checked(
        direct_semantic_image_preflight_command(image_id),
        timeout=60,
        log=args.output / "direct-text-semantics/image-preflight.log",
    )
    stack_log = (args.output / "direct-text-semantics/stack.log").open("wb")
    stack = subprocess.Popen(
        up_command(args), cwd=REPO_ROOT, stdout=stack_log, stderr=subprocess.STDOUT
    )
    try:
        wait_for_playground(stack, args.pipecat_port, args.startup_timeout)
        wait_model_idle(args.model_port)
        run_checked(
            direct_semantic_baseline_command(args),
            log=args.output / "direct-text-semantics/pocket.log",
            timeout=3600,
        )
        wait_model_idle(args.model_port)
    finally:
        if stack.poll() is None:
            stack.send_signal(signal.SIGINT)
            try:
                stack.wait(timeout=90)
            except subprocess.TimeoutExpired:
                stack.kill()
                stack.wait(timeout=10)
        stack_log.close()
    comparison = run_direct_semantic_gpu_ladder(args, image_id=image_id)
    report = {
        "passed": comparison.get("passed") is True,
        "mode": "direct_text_semantics_only",
        "direct_text_semantics": comparison,
    }
    atomic_json(args.output / "report.json", report)
    return report


def run_direct_function_logit_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run an untraced/traced/untraced W8 neutrality experiment."""

    args.output.mkdir(parents=True, exist_ok=False)
    args.direct_semantic_corpus = args.direct_semantic_corpus.expanduser().resolve()
    image_id = runtime_image_id(args.runtime_image)
    run_checked(
        direct_semantic_image_preflight_command(image_id),
        timeout=60,
        log=args.output / "image-preflight.log",
    )
    token_forms = ("text",)
    reports: dict[str, dict[str, Any]] = {}
    returncodes: dict[str, int] = {}
    for name, top_k in (
        ("off-a", 0),
        ("on", args.function_logit_trace_top_k),
        ("off-b", 0),
    ):
        run_dir = args.output / name
        run_dir.mkdir()
        command = direct_semantic_probe_command(
            args,
            precision="w8",
            runtime_image_id=image_id,
            output_relative=Path(name),
            token_forms=token_forms,
            function_logit_trace_top_k=top_k,
        )
        report, returncode = run_direct_semantic_probe_stage(
            command,
            log=args.output / f"{name}.log",
            report_path=run_dir / "report.json",
            precision="w8",
            runtime_image_id=image_id,
            corpus_path=args.direct_semantic_corpus,
            timeout=3600,
            expected_token_forms=token_forms,
            expected_trace_top_k=top_k,
        )
        reports[name] = report
        returncodes[name] = returncode
    neutrality = validate_function_logit_trace_neutrality(
        reports["off-a"], reports["on"], reports["off-b"]
    )
    result = {
        "schema": 1,
        "kind": "direct_function_logit_diagnostic",
        "passed": neutrality["passed"],
        "runtime_image_id": image_id,
        "corpus_sha256": corpus_sha256(args.direct_semantic_corpus),
        "token_forms": list(token_forms),
        "top_k": args.function_logit_trace_top_k,
        "trace_neutrality_within_observed_null": neutrality["passed"],
        "neutrality": neutrality,
        "returncodes": returncodes,
        "reports": {name: str((args.output / name / "report.json").resolve()) for name in reports},
    }
    atomic_json(args.output / "report.json", result)
    return result


def run_tool_freshness_parity_only(args: argparse.Namespace) -> dict[str, Any]:
    """Run the exact typed-vs-speech tool-freshness promotion experiment."""

    config = load_config()
    configured_image = config["image"]["runtime"]
    if args.runtime_image != configured_image:
        raise RuntimeError(
            "tool freshness parity must use the configured runtime image: "
            f"{args.runtime_image} != {configured_image}"
        )
    if args.asr_evaluator_image_id != PROMOTED_ASR_EVALUATOR_IMAGE_ID:
        raise RuntimeError("tool freshness parity requires the promoted ASR evaluator image")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime_id = runtime_image_id(args.runtime_image)
    state_path = args.output / "orchestration.json"
    state: dict[str, Any] = {
        "schema": 1,
        "kind": "tool_freshness_parity_orchestration",
        "passed": False,
        "runtime_image": args.runtime_image,
        "runtime_image_id": runtime_id,
        "asr_evaluator_image_id": args.asr_evaluator_image_id,
        "stages": [],
    }

    def stage(name: str, status: str, **evidence: Any) -> None:
        if status == "running":
            state["stages"].append({"name": name, "status": status, **evidence})
        else:
            if not state["stages"] or state["stages"][-1]["name"] != name:
                raise RuntimeError(f"tool freshness stage completion is out of order: {name}")
            state["stages"][-1].update({"status": status, **evidence})
        atomic_json(state_path, state)

    fixture_dir = args.output / "fixtures"
    fixture_dir.mkdir()
    fixture_command = tool_freshness_fixture_command(args, runtime_image_id=runtime_id)
    stage("fixture", "running", command=fixture_command)
    run_checked(
        fixture_command,
        log=args.output / "fixture.log",
        timeout=900,
    )
    scenario_sha256 = tool_freshness_canonical_sha256(scenario_document())
    manifest, _fixture_pcm = validate_tool_freshness_fixtures(
        fixture_dir,
        expected_runtime_image_id=runtime_id,
        expected_scenario_sha256=scenario_sha256,
    )
    validate_fixture_source_provenance(manifest, project_root=REPO_ROOT)
    stage(
        "fixture",
        "completed",
        manifest=str((fixture_dir / "manifest.json").resolve()),
        manifest_sha256=manifest["sha256"],
        scenario_sha256=scenario_sha256,
    )

    stack_log = (args.output / "stack.log").open("wb")
    stack_started = time.monotonic()
    stack: subprocess.Popen[bytes] = subprocess.Popen(
        up_command(args, trace_fc_async_heartbeat=True),
        cwd=REPO_ROOT,
        stdout=stack_log,
        stderr=subprocess.STDOUT,
    )
    stage("runtime_arms", "running", execution_order=["typed", "speech"])
    try:
        wait_for_playground(stack, args.pipecat_port, args.startup_timeout)
        ready_seconds = time.monotonic() - stack_started
        if ready_seconds > args.ready_budget_seconds:
            raise RuntimeError(
                f"stack readiness {ready_seconds:.3f}s exceeded "
                f"{args.ready_budget_seconds:.3f}s budget"
            )
        wait_model_idle(args.model_port)
        arm_command = tool_freshness_arm_command(args, runtime_image_id=runtime_id)
        run_checked(
            arm_command,
            log=args.output / "runtime.log",
            timeout=1200,
        )
        runtime_report_path = args.output / "runtime/report.json"
        runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
        if runtime_report.get("passed") is not True:
            raise RuntimeError("tool freshness runtime arms failed")
        stage(
            "runtime_arms",
            "completed",
            command=arm_command,
            report=str(runtime_report_path.resolve()),
            report_sha256=hashlib.sha256(runtime_report_path.read_bytes()).hexdigest(),
            cold_start_to_ready_seconds=round(ready_seconds, 3),
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        thread_dump: dict[str, Any]
        try:
            thread_dump = request_model_thread_dump(
                container_name=config["deployment"]["container_name"],
                stack_log_path=args.output / "stack.log",
                stack_log=stack_log,
            )
        except Exception as dump_exc:
            thread_dump = {
                "command": None,
                "returncode": None,
                "dump_observed": False,
                "error_type": type(dump_exc).__name__,
            }
        stage(
            "runtime_arms",
            "failed",
            error_type=type(exc).__name__,
            thread_dump=thread_dump,
        )
        raise
    finally:
        if stack.poll() is None:
            stack.send_signal(signal.SIGINT)
            try:
                stack.wait(timeout=90)
            except subprocess.TimeoutExpired:
                stack.kill()
                stack.wait(timeout=10)
        stack_log.close()

    asr_reports: dict[str, dict[str, Any]] = {}
    asr_returncodes: dict[str, int] = {}
    for modality in ("typed", "speech"):
        evaluated_dir = args.output / "asr" / modality
        evaluated_dir.mkdir(parents=True)
        command, log = asr_command(
            args,
            args.output / "runtime" / modality,
            f"{modality}-asr.log",
            max_silent_rate=0.0,
            output_dir=evaluated_dir,
        )
        stage(f"asr_{modality}", "running", command=command)
        report, returncode = run_json_report_command(
            command,
            log=log,
            report_path=evaluated_dir / "report.json",
            timeout=1800,
        )
        asr_reports[modality] = report
        asr_returncodes[modality] = returncode
        stage(
            f"asr_{modality}",
            "completed",
            returncode=returncode,
            passed=report["passed"],
            report=str((evaluated_dir / "report.json").resolve()),
            report_sha256=hashlib.sha256(
                (evaluated_dir / "report.json").read_bytes()
            ).hexdigest(),
        )

    compare_command = tool_freshness_compare_command(args)
    stage("compare", "running", command=compare_command)
    comparison_path = args.output / "comparison.json"
    comparison, comparison_returncode = run_json_report_command(
        compare_command,
        log=args.output / "comparison.log",
        report_path=comparison_path,
        timeout=300,
    )
    stage(
        "compare",
        "completed",
        returncode=comparison_returncode,
        passed=comparison["passed"],
        report=str(comparison_path.resolve()),
        report_sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
    )
    state["passed"] = comparison["passed"]
    atomic_json(state_path, state)
    report = {
        "schema": 1,
        "kind": "tool_freshness_parity_live_suite",
        "mode": "tool_freshness_parity_only",
        "passed": comparison["passed"],
        "runtime_image_id": runtime_id,
        "asr_evaluator_image_id": args.asr_evaluator_image_id,
        "fixture_manifest_sha256": manifest["sha256"],
        "scenario_sha256": scenario_sha256,
        "runtime_report": runtime_report,
        "asr_passed": {
            modality: asr_reports[modality]["passed"]
            for modality in ("typed", "speech")
        },
        "asr_returncodes": asr_returncodes,
        "comparison": comparison,
        "orchestration": {
            "path": str(state_path.resolve()),
            "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        },
    }
    atomic_json(args.output / "report.json", report)
    return report


def run_tool_freshness_reverse_diagnostic_only(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the preregistered, non-promotable speech-then-typed diagnostic."""

    config = load_config()
    configured_image = config["image"]["runtime"]
    if args.runtime_image != configured_image:
        raise RuntimeError(
            "tool freshness reverse diagnostic must use the configured runtime image"
        )
    if args.asr_evaluator_image_id != PROMOTED_ASR_EVALUATOR_IMAGE_ID:
        raise RuntimeError("reverse diagnostic requires the promoted ASR evaluator")
    preflight_path = args.output / "preflight.json"
    if args.output.exists():
        if (
            getattr(args, "_reverse_preflight_verified_in_process", False) is not True
            or not preflight_path.is_file()
        ):
            raise RuntimeError("reverse diagnostic preflight was not freshly verified")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight_returncode = 0
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        preflight_command = tool_freshness_reverse_analyzer_command(
            args, preflight_only=True
        )
        preflight, preflight_returncode = run_diagnostic_report_command(
            preflight_command,
            log=args.output / "preflight.log",
            report_path=preflight_path,
            timeout=300,
        )
    if (
        preflight_returncode != 0
        or preflight.get("diagnostic_complete") is not True
        or preflight.get("promotion_eligible") is not False
        or preflight.get("passed") is not False
    ):
        raise RuntimeError("frozen attempt-4 preflight failed")

    runtime_id = runtime_image_id(args.runtime_image)
    state_path = args.output / "orchestration.json"
    state: dict[str, Any] = {
        "schema": 1,
        "kind": "tool_freshness_reverse_order_diagnostic_orchestration",
        "passed": False,
        "promotion_eligible": False,
        "diagnostic_complete": False,
        "execution_order": ["speech", "typed"],
        "runtime_image": args.runtime_image,
        "runtime_image_id": runtime_id,
        "asr_evaluator_image_id": args.asr_evaluator_image_id,
        "preflight": {
            "path": str((args.output / "preflight.json").resolve()),
            "sha256": hashlib.sha256(
                preflight_path.read_bytes()
            ).hexdigest(),
        },
        "stages": [],
    }

    def stage(name: str, status: str, **evidence: Any) -> None:
        if status == "running":
            state["stages"].append({"name": name, "status": status, **evidence})
        else:
            if not state["stages"] or state["stages"][-1]["name"] != name:
                raise RuntimeError(f"reverse diagnostic stage is out of order: {name}")
            state["stages"][-1].update({"status": status, **evidence})
        atomic_json(state_path, state)

    fixture_dir = args.output / "fixtures"
    fixture_dir.mkdir()
    fixture_command = tool_freshness_fixture_command(args, runtime_image_id=runtime_id)
    stage("fixture", "running", command=fixture_command)
    run_checked(fixture_command, log=args.output / "fixture.log", timeout=900)
    scenario_sha256 = tool_freshness_canonical_sha256(scenario_document())
    manifest, _fixture_pcm = validate_tool_freshness_fixtures(
        fixture_dir,
        expected_runtime_image_id=runtime_id,
        expected_scenario_sha256=scenario_sha256,
    )
    validate_fixture_source_provenance(manifest, project_root=REPO_ROOT)
    stage(
        "fixture",
        "completed",
        manifest=str((fixture_dir / "manifest.json").resolve()),
        manifest_sha256=manifest["sha256"],
        scenario_sha256=scenario_sha256,
    )

    stack_log = (args.output / "stack.log").open("wb")
    stack_started = time.monotonic()
    stack: subprocess.Popen[bytes] = subprocess.Popen(
        up_command(args, trace_fc_async_heartbeat=True),
        cwd=REPO_ROOT,
        stdout=stack_log,
        stderr=subprocess.STDOUT,
    )
    stage("runtime_arms", "running", execution_order=["speech", "typed"])
    try:
        wait_for_playground(stack, args.pipecat_port, args.startup_timeout)
        ready_seconds = time.monotonic() - stack_started
        if ready_seconds > args.ready_budget_seconds:
            raise RuntimeError(
                f"stack readiness {ready_seconds:.3f}s exceeded "
                f"{args.ready_budget_seconds:.3f}s budget"
            )
        wait_model_idle(args.model_port)
        arm_command = tool_freshness_arm_command(
            args,
            runtime_image_id=runtime_id,
            nonpromotable_reverse_order=True,
        )
        run_checked(arm_command, log=args.output / "runtime.log", timeout=1200)
        runtime_report_path = args.output / "runtime/report.json"
        runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
        if (
            runtime_report.get("passed") is not True
            or runtime_report.get("promotion_eligible") is not False
            or runtime_report.get("execution_order") != ["speech", "typed"]
        ):
            raise RuntimeError("reverse diagnostic runtime arms failed")
        stage(
            "runtime_arms",
            "completed",
            command=arm_command,
            report=str(runtime_report_path.resolve()),
            report_sha256=hashlib.sha256(runtime_report_path.read_bytes()).hexdigest(),
            cold_start_to_ready_seconds=round(ready_seconds, 3),
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        try:
            thread_dump = request_model_thread_dump(
                container_name=config["deployment"]["container_name"],
                stack_log_path=args.output / "stack.log",
                stack_log=stack_log,
            )
        except Exception as dump_exc:
            thread_dump = {
                "command": None,
                "returncode": None,
                "dump_observed": False,
                "error_type": type(dump_exc).__name__,
            }
        stage(
            "runtime_arms",
            "failed",
            error_type=type(exc).__name__,
            thread_dump=thread_dump,
        )
        raise
    finally:
        if stack.poll() is None:
            stack.send_signal(signal.SIGINT)
            try:
                stack.wait(timeout=90)
            except subprocess.TimeoutExpired:
                stack.kill()
                stack.wait(timeout=10)
        stack_log.close()

    asr_reports: dict[str, dict[str, Any]] = {}
    asr_returncodes: dict[str, int] = {}
    for modality in ("speech", "typed"):
        evaluated_dir = args.output / "asr" / modality
        evaluated_dir.mkdir(parents=True)
        command, log = asr_command(
            args,
            args.output / "runtime" / modality,
            f"{modality}-asr.log",
            max_silent_rate=0.0,
            output_dir=evaluated_dir,
        )
        stage(f"asr_{modality}", "running", command=command)
        report, returncode = run_json_report_command(
            command,
            log=log,
            report_path=evaluated_dir / "report.json",
            timeout=1800,
        )
        if returncode != 0 or report.get("passed") is not True:
            raise RuntimeError(f"reverse diagnostic {modality} ASR integrity failed")
        asr_reports[modality] = report
        asr_returncodes[modality] = returncode
        stage(
            f"asr_{modality}",
            "completed",
            returncode=returncode,
            passed=True,
            report=str((evaluated_dir / "report.json").resolve()),
            report_sha256=hashlib.sha256(
                (evaluated_dir / "report.json").read_bytes()
            ).hexdigest(),
        )

    analyzer_command = tool_freshness_reverse_analyzer_command(args)
    stage("analyze", "running", command=analyzer_command)
    diagnostic_path = args.output / "diagnostic.json"
    diagnostic, diagnostic_returncode = run_diagnostic_report_command(
        analyzer_command,
        log=args.output / "diagnostic.log",
        report_path=diagnostic_path,
        timeout=300,
    )
    diagnostic_complete = (
        diagnostic_returncode == 0
        and diagnostic.get("diagnostic_complete") is True
        and diagnostic.get("passed") is False
        and diagnostic.get("promotion_eligible") is False
    )
    stage(
        "analyze",
        "completed",
        returncode=diagnostic_returncode,
        diagnostic_complete=diagnostic_complete,
        report=str(diagnostic_path.resolve()),
        report_sha256=hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
    )
    state["diagnostic_complete"] = diagnostic_complete
    atomic_json(state_path, state)
    report = {
        "schema": 1,
        "kind": "tool_freshness_reverse_order_diagnostic_live_suite",
        "mode": "tool_freshness_reverse_diagnostic_only",
        "passed": False,
        "promotion_eligible": False,
        "diagnostic_complete": diagnostic_complete,
        "execution_order": ["speech", "typed"],
        "runtime_image_id": runtime_id,
        "asr_evaluator_image_id": args.asr_evaluator_image_id,
        "fixture_manifest_sha256": manifest["sha256"],
        "scenario_sha256": scenario_sha256,
        "runtime_report": runtime_report,
        "asr_passed": {
            modality: asr_reports[modality]["passed"]
            for modality in ("speech", "typed")
        },
        "asr_returncodes": asr_returncodes,
        "diagnostic": diagnostic,
        "orchestration": {
            "path": str(state_path.resolve()),
            "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        },
    }
    atomic_json(args.output / "report.json", report)
    return report


def run_component_gates(args: argparse.Namespace) -> None:
    components = args.output / "components"
    components.mkdir(parents=True, exist_ok=True)
    corpus = args.cache_root / "qualification-corpus"
    extract_replay_corpus(args.cache_root / "artifacts" / "release", corpus)
    prefix = gpu_container_prefix(args)
    run_checked(
        [
            *prefix,
            "-v",
            f"{corpus}:/corpus:ro",
            "-v",
            f"{components}:/qualification",
            args.runtime_image,
            "python3",
            "/workspace/project/tools/conversion/nano_component_gate.py",
            "--speech-root",
            "/opt/Speech",
            "--nano-vllm-path",
            "/models/derived/nano",
            "--teacher-replay",
            "/corpus/evaluation/0",
            "--output-dir",
            "/qualification/nano",
            "--max-calls",
            "1172",
        ],
        log=args.output / "nano-component.log",
        timeout=3600,
    )
    for name, graph_arguments in (
        ("eartts-eager", []),
        ("eartts-graph", ["--no-enforce-eager"]),
    ):
        run_checked(
            [
                *prefix,
                "-v",
                f"{components}:/qualification",
                args.runtime_image,
                "python3",
                "/workspace/project/tools/conversion/eartts_component_gate.py",
                "--speech-root",
                "/opt/Speech",
                "--checkpoint-root",
                "/models/voicechat",
                "--nano-skeleton",
                "/models/NVIDIA-Nemotron-Nano-9B-v2",
                "--eartts-vllm-path",
                "/models/derived/eartts",
                "--output-dir",
                f"/qualification/{name}",
                *graph_arguments,
            ],
            log=args.output / f"{name}-component.log",
            timeout=3600,
        )
        run_checked(
            eartts_asr_command(args, components, name),
            log=args.output / f"{name}-asr.log",
            timeout=1800,
        )
    run_checked(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-e",
            "HF_HOME=/models/huggingface",
            "-e",
            "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
            "-v",
            f"{REPO_ROOT}:/workspace/project:ro",
            "-v",
            f"{args.cache_root / 'huggingface'}:/models/huggingface:ro",
            "-v",
            f"{components}:/qualification",
            args.runtime_image,
            "python3",
            "/workspace/project/tools/qualification/pocket_worker_gate.py",
            "--output",
            "/qualification/pocket",
        ],
        log=args.output / "pocket-component.log",
        timeout=900,
    )


def run_conversion_tensor_tests(args: argparse.Namespace) -> None:
    """Run torch-dependent conversion contracts in the pinned CUDA image.

    The Pipecat host environment deliberately has no PyTorch. These tests are
    collected as visible skips there and executed here before any GPU gate.
    """

    run_checked(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--workdir",
            "/workspace/project",
            "-v",
            f"{REPO_ROOT}:/workspace/project:ro",
            args.runtime_image,
            "python3",
            "-m",
            "pytest",
            "-q",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "tests/conversion/test_nano_attribution.py",
            "tests/conversion/test_nano_gptq_calibration.py",
            "tests/conversion/test_package_replay_corpus.py",
            "tests/conversion/test_nano_replay.py",
            "tests/conversion/test_eartts_component_gate.py",
            (
                "tests/runtime/test_pocket_worker_ipc.py::"
                "test_internal_text_carrier_is_deterministic_per_text_and_restores_rng"
            ),
        ],
        log=args.output / "conversion-tensor-tests.log",
        timeout=300,
    )


def prepare_fixtures(args: argparse.Namespace) -> None:
    if args.fixture_manifest is None:
        run_checked(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-e",
                "HF_HOME=/models/huggingface",
                "-e",
                "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
                "-v",
                f"{REPO_ROOT}:/workspace/project:ro",
                "-v",
                f"{args.cache_root / 'huggingface'}:/models/huggingface:ro",
                "-v",
                f"{args.output}:/qualification",
                args.runtime_image,
                "python3",
                "/workspace/project/tools/qualification/generate_multiturn_fixtures.py",
                "--output",
                "/qualification/fixtures",
            ],
            log=args.output / "fixture-generation.log",
            timeout=900,
        )
        generated_manifest = args.output / "fixtures/manifest.json"
        generated = json.loads(generated_manifest.read_text(encoding="utf-8"))
        for turn in generated["turns"]:
            turn["audio_path"] = str(generated_manifest.parent / Path(turn["audio_path"]).name)
        for case in generated.get("memory_matrix", {}).get("cases", []):
            for field in ("seed_audio_path", "recall_audio_path"):
                if field in case:
                    case[field] = str(generated_manifest.parent / Path(case[field]).name)
        generated["browser_microphone_wav"] = str(
            generated_manifest.parent / Path(generated["browser_microphone_wav"]).name
        )
        # Preserve the container-produced manifest as immutable evidence. The
        # host owns the run root and writes its path-remapped view alongside it.
        args.fixture_manifest = args.output / "fixture-manifest-host.json"
        args.fixture_manifest.write_text(
            json.dumps(generated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    if args.browser_mic_wav is None:
        configured = fixture.get("browser_microphone_wav")
        args.browser_mic_wav = (
            Path(configured) if configured else Path(fixture["turns"][0]["audio_path"])
        )


def require_browser_executable() -> Path:
    """Fail before GPU work when Playwright's pinned Chromium is absent."""

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        executable = Path(playwright.chromium.executable_path)
    if not executable.is_file():
        raise RuntimeError(
            f"Playwright Chromium is not installed at {executable}; run "
            "`uv run --frozen playwright install chromium` before the live suite"
        )
    return executable


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "direct_function_logit_diagnostic", False):
        return run_direct_function_logit_diagnostic(args)
    if getattr(args, "tool_freshness_parity_only", False):
        return run_tool_freshness_parity_only(args)
    if getattr(args, "tool_freshness_reverse_diagnostic_only", False):
        return run_tool_freshness_reverse_diagnostic_only(args)
    if args.direct_text_semantics_only:
        return run_direct_semantics_only(args)
    chromium = require_browser_executable()
    args.output.mkdir(parents=True, exist_ok=False)
    run_conversion_tensor_tests(args)
    prepare_fixtures(args)
    run_component_gates(args)
    if args.restart_cycles:
        run_checked(
            [
                str(args.python),
                str(REPO_ROOT / "tools/qualification/restart_cycle_gate.py"),
                "--cache-root",
                str(args.cache_root),
                "--trace-root",
                str(args.trace_root),
                "--output",
                str(args.output / "restart-cycles"),
                "--cycles",
                str(args.restart_cycles),
                "--model-port",
                str(args.model_port),
                "--pipecat-port",
                str(args.pipecat_port),
                "--ready-budget-seconds",
                str(args.ready_budget_seconds),
            ],
            timeout=(args.restart_cycles + 2) * (args.startup_timeout + 120),
        )

    stack_log = (args.output / "stack.log").open("wb")
    stack_started = time.monotonic()
    stack: subprocess.Popen[bytes] = subprocess.Popen(
        up_command(args), cwd=REPO_ROOT, stdout=stack_log, stderr=subprocess.STDOUT
    )
    try:
        wait_for_playground(stack, args.pipecat_port, args.startup_timeout)
        stack_ready_seconds = time.monotonic() - stack_started
        if stack_ready_seconds > args.ready_budget_seconds:
            raise RuntimeError(
                f"stack readiness {stack_ready_seconds:.3f}s exceeded "
                f"{args.ready_budget_seconds:.3f}s budget"
            )
        wait_model_idle(args.model_port)
        browser_environment = os.environ.copy()
        browser_environment.update(
            {
                "VOICECHAT_LIVE_PLAYGROUND_URL": f"http://127.0.0.1:{args.pipecat_port}/client/",
                "VOICECHAT_LIVE_MIC_WAV": str(args.browser_mic_wav),
                "PLAYWRIGHT_CHROMIUM_EXECUTABLE": str(chromium),
            }
        )
        browser_report = run_browser_gate(args, browser_environment)
        wait_model_idle(args.model_port)
        run_checked(
            [
                str(args.python),
                str(REPO_ROOT / "tools/qualification/multiturn_strict_v3.py"),
                "--url",
                f"ws://127.0.0.1:{args.model_port}/v1/realtime",
                "--manifest",
                str(args.fixture_manifest),
                "--output",
                str(args.output / "multiturn"),
            ],
            log=args.output / "multiturn.log",
            timeout=900,
        )
        wait_model_idle(args.model_port)
        run_checked(
            [
                str(args.python),
                str(REPO_ROOT / "tools/qualification/memory_matrix_strict_v3.py"),
                "--url",
                f"ws://127.0.0.1:{args.model_port}/v1/realtime",
                "--manifest",
                str(args.fixture_manifest),
                "--output",
                str(args.output / "memory-matrix"),
            ],
            log=args.output / "memory-matrix.log",
            timeout=1800,
        )
        wait_model_idle(args.model_port)
        sustained_command = [
            str(args.python),
            str(REPO_ROOT / "tools/qualification/sustained_strict_v3.py"),
            "--url",
            f"ws://127.0.0.1:{args.model_port}/v1/realtime",
            "--output",
            str(args.output / "sustained"),
            "--duration-seconds",
            str(args.duration_seconds),
        ]
        sustained_command.append(
            "--expect-session-limit" if args.expect_session_limit else "--no-expect-session-limit"
        )
        run_checked(
            sustained_command,
            log=args.output / "sustained.log",
            timeout=args.duration_seconds + 900,
        )
    finally:
        if stack.poll() is None:
            stack.send_signal(signal.SIGINT)
            try:
                stack.wait(timeout=90)
            except subprocess.TimeoutExpired:
                stack.kill()
                stack.wait(timeout=10)
        stack_log.close()

    for run_dir, report_name, max_silent_rate in (
        (args.output / "multiturn", "multiturn-asr.log", None),
        (args.output / "memory-matrix", "memory-matrix-asr.log", 0.0),
        (args.output / "sustained", "sustained-asr.log", None),
    ):
        command, report_log = asr_command(
            args, run_dir, report_name, max_silent_rate=max_silent_rate
        )
        run_checked(command, log=report_log, timeout=1800)

    multiturn_report = json.loads((args.output / "multiturn/report.json").read_text())
    memory_matrix_report = json.loads((args.output / "memory-matrix/report.json").read_text())
    sustained_report = json.loads((args.output / "sustained/report.json").read_text())
    report = {
        "passed": bool(
            browser_report["passed"]
            and multiturn_report["passed"]
            and memory_matrix_report["passed"]
            and sustained_report["passed"]
        ),
        "browser": browser_report,
        "multiturn": multiturn_report,
        "memory_matrix": memory_matrix_report,
        "sustained": sustained_report,
        "restart_cycles": args.restart_cycles,
        "cold_start_to_playground_ready_seconds": round(stack_ready_seconds, 3),
        "cold_start_budget_seconds": args.ready_budget_seconds,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--browser-mic-wav", type=Path)
    parser.add_argument("--asr-evaluator-image", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--direct-text-semantics-only", action="store_true")
    parser.add_argument("--direct-function-logit-diagnostic", action="store_true")
    parser.add_argument("--tool-freshness-parity-only", action="store_true")
    parser.add_argument(
        "--tool-freshness-reverse-diagnostic-only", action="store_true"
    )
    parser.add_argument(
        "--reverse-forward-comparison",
        type=Path,
        default=Path(
            "/home/khkramer/.cache/nemotron-voicechat/traces/"
            "tool-freshness-parity-20260809-attempt4/comparison.json"
        ),
    )
    parser.add_argument(
        "--reverse-forward-fixture",
        type=Path,
        default=Path(
            "/home/khkramer/.cache/nemotron-voicechat/traces/"
            "tool-freshness-parity-20260809-attempt4/fixtures/manifest.json"
        ),
    )
    parser.add_argument("--function-logit-trace-top-k", type=int, default=10)
    parser.add_argument("--diagnostic-speech-root", type=Path)
    parser.add_argument("--fp32-derived-root", type=Path)
    parser.add_argument(
        "--direct-semantic-corpus",
        type=Path,
        default=REPO_ROOT / "tools/qualification/direct_text_semantic_corpus.json",
    )
    parser.add_argument("--duration-seconds", type=float, default=1200)
    parser.add_argument("--restart-cycles", type=int, default=3)
    parser.add_argument("--model-port", type=int, default=8786)
    parser.add_argument("--pipecat-port", type=int, default=7860)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--ready-budget-seconds", type=float, required=True)
    parser.add_argument(
        "--expect-session-limit", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    diagnostic_mode = args.tool_freshness_reverse_diagnostic_only
    try:
        exclusive_modes = (
            args.direct_text_semantics_only,
            args.direct_function_logit_diagnostic,
            args.tool_freshness_parity_only,
            diagnostic_mode,
        )
        if sum(bool(mode) for mode in exclusive_modes) > 1:
            raise ValueError("standalone qualification modes are mutually exclusive")
        args.asr_evaluator_image_id = runtime_image_id(args.asr_evaluator_image)
        if diagnostic_mode:
            args.output.mkdir(parents=True, exist_ok=False)
            preflight_command = tool_freshness_reverse_analyzer_command(
                args, preflight_only=True
            )
            preflight, returncode = run_diagnostic_report_command(
                preflight_command,
                log=args.output / "preflight.log",
                report_path=args.output / "preflight.json",
                timeout=300,
            )
            if (
                returncode != 0
                or preflight.get("diagnostic_complete") is not True
                or preflight.get("passed") is not False
                or preflight.get("promotion_eligible") is not False
            ):
                raise RuntimeError("frozen attempt-4 preflight failed before GPU work")
            args._reverse_preflight_verified_in_process = True
        subprocess.run(
            [
                str(REPO_ROOT / "container/audit-asr-evaluator.sh"),
                args.asr_evaluator_image_id,
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        report = run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        if diagnostic_mode:
            report.update(
                {
                    "promotion_eligible": False,
                    "diagnostic_complete": False,
                }
            )
        atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    successful = (
        report.get("diagnostic_complete") is True
        if diagnostic_mode
        else report.get("passed") is True
    )
    raise SystemExit(0 if successful else 1)


if __name__ == "__main__":
    main()
