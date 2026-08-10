"""Dependency-light contract for typed-versus-speech tool freshness parity."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from .tool_freshness_fixture import (
    SCENARIO_SHA256_V1,
    SCENARIO_SHA256_V2,
    TOOL_ONLY_PROMPTS,
    TOOL_ONLY_PROMPTS_V1,
    TOOL_RESULT_ORDINALS,
    TOOL_RESULT_TIMES_V1,
    TOOL_VERIFICATION_CODES_V1,
    TOOL_VERIFICATION_WORDS,
    canonical_sha256,
)

InputModality = Literal["typed", "speech"]
PARITY_MARGIN_POLICY = "parity_margin_v2"
MAX_INPUT_WER = 0.35
SESSION_INSTRUCTIONS = "Answer briefly. Use tools only when requested."
TRAILING_SILENCE_FRAMES = 3
STRUCTURAL_CONJUNCTS_V1 = (
    "scenario_identity",
    "runtime_fixture_identity",
    "response_cardinality",
    "input_lifecycle",
    "tool_calls",
    "response_correlation",
    "response_brackets",
    "eotr_valid",
    "function_channel_idle",
    "post_eotr_audible",
    "bounded_watchdog_close",
    "text_nonempty",
    "audio_nonempty",
    "cycle_cardinality",
    "protocol_errors_absent",
    "clean_session_close",
    "heard_prompt",
)
STRUCTURAL_CONJUNCTS_V2 = (
    "scenario_identity",
    "runtime_fixture_identity",
    "response_cardinality",
    "input_lifecycle",
    "tool_calls",
    "response_correlation",
    "response_brackets",
    "eotr_valid",
    "function_channel_idle",
    "post_eotr_audible",
    "bounded_watchdog_close",
    "text_nonempty",
    "audio_nonempty",
    "cycle_cardinality",
    "protocol_errors_absent",
    "clean_session_close",
    "heard_prompt",
)
STRUCTURAL_CONJUNCTS = STRUCTURAL_CONJUNCTS_V2
INPUT_EVENT_TYPES = (
    "input_text.accepted",
    "input_text.injection_started",
    "input_text.injection_finished",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.turn_started",
    "input_audio_buffer.committed",
    "conversation.item.input_audio_transcription.completed",
)
RESPONSE_SINGLETON_EVENT_TYPES = (
    "response.created",
    "response.output_text.done",
    "response.output_audio.done",
    "response.done",
    "response.function_call_arguments.done",
    "conversation.item.function_call_output.applied",
)


def contains_literal_token_sequence(text: str, candidate: str) -> bool:
    haystack = re.findall(r"[a-z0-9]+", (text or "").lower())
    needle = re.findall(r"[a-z0-9]+", (candidate or "").lower())
    return bool(needle) and any(
        haystack[offset : offset + len(needle)] == needle
        for offset in range(len(haystack) - len(needle) + 1)
    )


def tool_definition_v1(mode: str) -> dict[str, Any]:
    description = "Get the current clock time in UTC."
    if mode == "tool-only":
        description = (
            "Get the current clock time in UTC and a fresh verification code. "
            "Call it again whenever the user requests a new code."
        )
    return {
        "name": "get_current_utc_time",
        "description": description,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def tool_result_v1(
    mode: str, *, now: str, verification_code: str
) -> tuple[str, bool, dict[str, str]]:
    if mode == "tool-only":
        return (
            verification_code,
            True,
            {
                "timezone": "UTC",
                "spoken": f"{verification_code.capitalize()}. Current UTC time is {now}.",
            },
        )
    return now, False, {"timezone": "UTC", "spoken": now}


def scenario_document_v1() -> dict[str, Any]:
    outputs = []
    for now, code in zip(TOOL_RESULT_TIMES_V1, TOOL_VERIFICATION_CODES_V1, strict=True):
        expected, semantic_required, payload = tool_result_v1(
            "tool-only", now=now, verification_code=code
        )
        outputs.append(
            {
                "now": now,
                "verification_code": code,
                "expected_response": expected,
                "semantic_required": semantic_required,
                "function_output": payload,
            }
        )
    return {
        "instructions": SESSION_INSTRUCTIONS,
        "prompts": list(TOOL_ONLY_PROMPTS_V1),
        "tool": tool_definition_v1("tool-only"),
        "tool_outputs": outputs,
        "speech_transport": {
            "encoding": "pcm16",
            "sample_rate": 16_000,
            "channels": 1,
            "frame_samples": 1_280,
            "trailing_silence_frames": TRAILING_SILENCE_FRAMES,
            "turn_detection": "client_smart_turn_v1",
            "boundary": "input_audio_buffer.commit",
        },
    }


def tool_definition_v2() -> dict[str, Any]:
    return {
        "name": "get_fresh_verification_word",
        "description": (
            "Return one fresh verification word. Call it again whenever the user asks "
            "for a new word."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def tool_result_v2(
    *, verification_word: str, ordinal_word: str
) -> tuple[str, bool, dict[str, str]]:
    return (
        verification_word,
        True,
        {
            "spoken": (
                f"{verification_word.capitalize()} is the {ordinal_word} fresh word, ready now."
            ),
        },
    )


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z]+", value.lower())


def _reject_unmeasurable_utterance(value: str, *, label: str) -> None:
    tokens = re.findall(r"[A-Za-z0-9]+", value)
    if not value or any(character.isdigit() for character in value):
        raise RuntimeError(f"{label} contains a digit or is empty")
    if any(len(token) >= 2 and token.isupper() for token in tokens):
        raise RuntimeError(f"{label} contains an acronym")
    if any(
        token.lower()
        in {
            "utc",
            "timezone",
            "clock",
            "time",
            "times",
            "hour",
            "hours",
            "minute",
            "minutes",
        }
        for token in tokens
    ):
        raise RuntimeError(f"{label} contains time vocabulary")


def validate_measurable_scenario_v2(scenario: dict[str, Any]) -> None:
    """Fail closed if the ASR-scored corpus drifts from ordinary spoken English."""

    _reject_unmeasurable_utterance(
        str(scenario.get("instructions") or ""), label="scenario instructions"
    )
    tool = scenario.get("tool") or {}
    _reject_unmeasurable_utterance(str(tool.get("name") or "").replace("_", " "), label="tool name")
    _reject_unmeasurable_utterance(str(tool.get("description") or ""), label="tool description")
    prompts = scenario.get("prompts") or []
    outputs = scenario.get("tool_outputs") or []
    if len(prompts) != 4 or len(outputs) != 4:
        raise RuntimeError("scenario v2 cardinality is invalid")
    for ordinal, prompt in enumerate(prompts, 1):
        _reject_unmeasurable_utterance(str(prompt), label=f"prompt {ordinal}")
        if len(_normalized_words(str(prompt))) < 6:
            raise RuntimeError("scenario v2 prompt is too short for the heard gate")
    for ordinal, (output, word) in enumerate(zip(outputs, TOOL_VERIFICATION_WORDS, strict=True), 1):
        spoken = str((output.get("function_output") or {}).get("spoken") or "")
        _reject_unmeasurable_utterance(spoken, label=f"spoken output {ordinal}")
        words = _normalized_words(spoken)
        if (
            len(words) < 6
            or words[0] != word
            or len(re.findall(r"[.!?]", spoken)) != 1
            or not spoken.endswith(".")
        ):
            raise RuntimeError("scenario v2 spoken output is not one measurable sentence")
        if set(output.get("function_output") or {}) != {"spoken"}:
            raise RuntimeError("scenario v2 injects fields outside the spoken result")
        if output.get("expected_response") != word:
            raise RuntimeError("scenario v2 expected response drifted")
        if output.get("excluded_responses") != list(TOOL_VERIFICATION_WORDS[: ordinal - 1]):
            raise RuntimeError("scenario v2 stale-result exclusion drifted")


def scenario_document_v2() -> dict[str, Any]:
    outputs = []
    for ordinal, (ordinal_word, word) in enumerate(
        zip(TOOL_RESULT_ORDINALS, TOOL_VERIFICATION_WORDS, strict=True), 1
    ):
        expected, semantic_required, payload = tool_result_v2(
            verification_word=word, ordinal_word=ordinal_word
        )
        outputs.append(
            {
                "ordinal_word": ordinal_word,
                "verification_word": word,
                "expected_response": expected,
                "excluded_responses": list(TOOL_VERIFICATION_WORDS[: ordinal - 1]),
                "semantic_required": semantic_required,
                "function_output": payload,
            }
        )
    scenario = {
        "instructions": SESSION_INSTRUCTIONS,
        "prompts": list(TOOL_ONLY_PROMPTS),
        "tool": tool_definition_v2(),
        "tool_outputs": outputs,
        "speech_transport": {
            "encoding": "pcm16",
            "sample_rate": 16_000,
            "channels": 1,
            "frame_samples": 1_280,
            "trailing_silence_frames": TRAILING_SILENCE_FRAMES,
            "turn_detection": "client_smart_turn_v1",
            "boundary": "input_audio_buffer.commit",
        },
    }
    validate_measurable_scenario_v2(scenario)
    return scenario


def scenario_document_for_sha256(scenario_sha256: str) -> dict[str, Any]:
    if scenario_sha256 == SCENARIO_SHA256_V1:
        return scenario_document_v1()
    if scenario_sha256 == SCENARIO_SHA256_V2:
        return scenario_document_v2()
    raise RuntimeError("unknown tool freshness scenario hash")


def scenario_document() -> dict[str, Any]:
    """Return the active, preregistered ordinary-English scenario."""

    scenario = scenario_document_v2()
    if canonical_sha256(scenario) != SCENARIO_SHA256_V2:
        raise RuntimeError("active tool freshness scenario drifted from its pinned hash")
    return scenario


def tool_definition(mode: str) -> dict[str, Any]:
    return tool_definition_v2() if mode == "tool-only" else tool_definition_v1(mode)


def tool_result(
    mode: str,
    *,
    now: str | None = None,
    verification_word: str | None = None,
    ordinal_word: str | None = None,
) -> tuple[str, bool, dict[str, str]]:
    if mode == "tool-only":
        if not verification_word or not ordinal_word:
            raise ValueError("tool-only result requires verification and ordinal words")
        return tool_result_v2(verification_word=verification_word, ordinal_word=ordinal_word)
    if not now:
        raise ValueError("mixed result requires a clock value")
    return tool_result_v1(mode, now=now, verification_code="unused")


def health_synthesis_identity(health: dict[str, Any]) -> dict[str, Any]:
    return {
        key: health.get(key)
        for key in (
            "language",
            "voice",
            "threads",
            "cpus",
            "seed",
            "seed_scheme",
            "package_version",
            "assets",
        )
    }


def fixture_synthesis_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    synthesis = manifest["synthesis"]
    return {
        "language": synthesis["language"],
        "voice": synthesis["voice"],
        "threads": synthesis["threads"],
        "cpus": synthesis["cpus"],
        "seed": synthesis["base_seed"],
        "seed_scheme": synthesis["seed_scheme"],
        "package_version": synthesis["package_version"],
        "assets": synthesis["assets"],
    }


def experiment_sha256(*, scenario_sha256: str, fixture_manifest_sha256: str) -> str:
    return canonical_sha256(
        {
            "scenario_sha256": scenario_sha256,
            "fixture_manifest_sha256": fixture_manifest_sha256,
        }
    )


def structural_conjuncts_for_scenario(scenario: dict[str, Any]) -> tuple[str, ...]:
    scenario_sha256 = canonical_sha256(scenario)
    if scenario_sha256 == SCENARIO_SHA256_V1:
        return STRUCTURAL_CONJUNCTS_V1
    if scenario_sha256 == SCENARIO_SHA256_V2:
        return STRUCTURAL_CONJUNCTS_V2
    raise RuntimeError("unknown tool freshness scenario hash")


def evaluate_tool_response_channels(
    responses: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    eotr_observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    for response in responses:
        if not response.get("expected_response_any"):
            continue
        response_id = response.get("response_id")
        turn_id = response.get("turn_id")
        observations = [item for item in eotr_observations if item.get("turn_id") == turn_id]
        response_metrics = [item for item in metrics if item.get("response_id") == response_id]
        observation = observations[-1] if observations else None
        observation_transport_frame = observation.get("transport_frame") if observation else None
        post_eotr_metrics = [
            item
            for item in response_metrics
            if not isinstance(observation_transport_frame, int)
            or (
                isinstance(item.get("transport_frame"), int)
                and item["transport_frame"] > observation_transport_frame
            )
        ]
        nonpad_function_frames = [
            int(item["transport_frame"])
            for item in post_eotr_metrics
            if (
                ((item.get("turn_state") or {}).get("function_calling") or {}).get(
                    "effective_token_is_pad"
                )
                is False
            )
            and isinstance(item.get("transport_frame"), int)
        ]
        audible_post_eotr_frames = [
            int(item["transport_frame"])
            for item in post_eotr_metrics
            if isinstance(item.get("transport_frame"), int)
            and isinstance((item.get("output_audio") or {}).get("rms_dbfs"), (int, float))
            and float((item.get("output_audio") or {})["rms_dbfs"])
            > float(
                (
                    ((item.get("turn_state") or {}).get("agent_silence_watchdog") or {}).get(
                        "threshold_dbfs", -90.0
                    )
                )
            )
        ]
        eotr_valid = bool(
            observation
            and observation.get("observed") is True
            and observation.get("wait_steps") == 1
            and observation.get("timeout") is False
            and isinstance(observation.get("eotr_token_id"), int)
            and observation.get("first_post_drain_token_id") == observation.get("eotr_token_id")
            and observation.get("first_post_drain_token_frame") == observation.get("observed_frame")
            and observation.get("unexpected_token_id") is None
            and observation.get("unexpected_token_frame") is None
            and observation.get("feedback_token_id") == observation.get("eotr_token_id")
            and observation.get("feedback_frame") == observation.get("observed_frame")
            and observation.get("feedback_committed") is True
        )
        expected = response.get("expected_response_any") or []
        text_semantic_required = response.get("expected_text_semantic_required") is True
        text_semantic_match = bool(
            not text_semantic_required
            or any(
                contains_literal_token_sequence(str(response.get("text") or ""), candidate)
                for candidate in expected
                if isinstance(candidate, str)
            )
        )
        checks.append(
            {
                "turn_id": turn_id,
                "response_id": response_id,
                "eotr_observation": observation,
                "eotr_valid": eotr_valid,
                "function_channel_nonpad_frames": nonpad_function_frames,
                "function_channel_idle": not nonpad_function_frames,
                "audible_post_eotr_frames": audible_post_eotr_frames,
                "post_eotr_audible": bool(audible_post_eotr_frames),
                "bounded_watchdog_close": response.get("done_reason")
                in {"decoded_silence_watchdog", "no_text_since_bos_watchdog"},
                "done_reason": response.get("done_reason"),
                "text_nonempty": bool(str(response.get("text") or "").strip()),
                "text_semantic_required": text_semantic_required,
                "text_semantic_match": text_semantic_match,
                "audio_nonempty": int(response.get("audio_bytes") or 0) > 0,
            }
        )
    passed = bool(checks) and all(
        check["eotr_valid"]
        and check["function_channel_idle"]
        and check["post_eotr_audible"]
        and check["bounded_watchdog_close"]
        and check["text_nonempty"]
        and check["text_semantic_match"]
        and check["audio_nonempty"]
        for check in checks
    )
    return checks, passed


def evaluate_tool_cycle_cardinality(
    *,
    requested_tool_prompts: int,
    tool_response_checks: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    function_snapshots = [
        (item.get("turn_state") or {}).get("function_calling")
        for item in metrics
        if isinstance((item.get("turn_state") or {}).get("function_calling"), dict)
    ]
    committed_values = [
        item["client_eou_sotc_committed_count"]
        for item in function_snapshots
        if isinstance(item.get("client_eou_sotc_committed_count"), int)
    ]
    forced_values = [
        item["post_fc_client_bos_forced_count"]
        for item in function_snapshots
        if isinstance(item.get("post_fc_client_bos_forced_count"), int)
    ]
    eotr_observed_values = [
        item["eotr_observed_count"]
        for item in function_snapshots
        if isinstance(item.get("eotr_observed_count"), int)
    ]
    committed = max(committed_values) if committed_values else None
    forced = max(forced_values) if forced_values else None
    eotr_observed = max(eotr_observed_values) if eotr_observed_values else None
    checked = len(tool_response_checks)
    return {
        "passed": bool(
            requested_tool_prompts > 0
            and checked == requested_tool_prompts
            and committed == requested_tool_prompts
            and forced == requested_tool_prompts
            and eotr_observed == requested_tool_prompts
        ),
        "requested_tool_prompts": requested_tool_prompts,
        "validated_tool_responses": checked,
        "client_eou_sotc_committed_count": committed,
        "post_fc_client_bos_forced_count": forced,
        "eotr_observed_count": eotr_observed,
        "criterion": (
            "requested == validated == observed EOTR == boundary commits == post-FC forced BOS"
        ),
    }


def derive_structural_conjuncts(
    *,
    modality: InputModality,
    inputs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    cardinality: dict[str, Any],
    errors: list[dict[str, Any]],
    session_closed: dict[str, Any] | None,
    scenario: dict[str, Any],
    fixtures: list[dict[str, Any]],
    scenario_identity: bool,
    runtime_fixture_identity: bool,
) -> dict[str, bool]:
    prompts = scenario.get("prompts") or []
    outputs = scenario.get("tool_outputs") or []
    scenario_turn_identity = bool(
        len(prompts) == len(outputs) == len(fixtures) == 4
        and len(inputs) == 4
        and all(
            fixture.get("case_id") == f"tool-freshness-{ordinal}"
            and fixture.get("ordinal") == ordinal
            and fixture.get("text") == prompt
            and item.get("case_id") == fixture.get("case_id")
            and item.get("ordinal") == ordinal
            and item.get("input_modality") == modality
            and item.get("expected_input_text") == prompt
            for ordinal, (prompt, fixture, item) in enumerate(
                zip(prompts, fixtures, inputs, strict=True), 1
            )
        )
    )
    input_turn_ids = [item.get("turn_id") for item in inputs]
    input_turn_identity = bool(
        len(set(input_turn_ids)) == 4
        and all(isinstance(turn_id, str) and turn_id for turn_id in input_turn_ids)
    )
    input_lifecycle = (
        len(inputs) == 4
        and all(
            (
                item.get("accepted") is True
                and item.get("started") is True
                and item.get("disposition") == "completed"
                and item.get("requested_text") == item.get("expected_input_text")
                and item.get("speech_started") is True
                and item.get("transcription_source") == "typed"
                and item.get("transcription_job_id") == item.get("job_id")
                and item.get("transcription_turn_id") == item.get("turn_id")
                and item.get("event_counts")
                == {
                    "input_text.accepted": 1,
                    "input_text.injection_started": 1,
                    "input_text.injection_finished": 1,
                    "input_audio_buffer.speech_started": 1,
                    "conversation.item.input_audio_transcription.completed": 1,
                }
                and len(item.get("event_correlations") or []) >= 5
                and all(item.get("event_correlations") or [])
            )
            if modality == "typed"
            else (
                item.get("speech_started") is True
                and item.get("turn_started_ack") is True
                and item.get("committed_ack") is True
                and item.get("client_turn_id") == item.get("ordinal")
                and item.get("turn_started_turn_id") == item.get("turn_id")
                and item.get("committed_turn_id") == item.get("turn_id")
                and item.get("transcription_source") == "microphone"
                and item.get("transcription_turn_id") == item.get("turn_id")
                and bool(str(item.get("displayed_input_text") or "").strip())
                and item.get("event_counts")
                == {
                    "input_audio_buffer.speech_started": 1,
                    "input_audio_buffer.turn_started": 1,
                    "input_audio_buffer.committed": 1,
                    "conversation.item.input_audio_transcription.completed": 1,
                }
                and len(item.get("event_correlations") or []) >= 4
                and all(item.get("event_correlations") or [])
            )
            for item in inputs
        )
        and input_turn_identity
    )
    response_ids = [response.get("response_id") for response in responses]
    response_correlation = (
        len(responses) == 4
        and len(set(response_ids)) == 4
        and all(isinstance(response_id, str) and response_id for response_id in response_ids)
        and all(
            response.get("turn_id") == input_item.get("turn_id")
            and response.get("case_id") == input_item.get("case_id")
            and response.get("ordinal") == input_item.get("ordinal")
            and response.get("protocol_correlation") is True
            and response.get("event_counts") == {kind: 1 for kind in RESPONSE_SINGLETON_EVENT_TYPES}
            and len(response.get("event_correlations") or []) >= 6
            and all(response.get("event_correlations") or [])
            for response, input_item in zip(responses, inputs, strict=True)
        )
    )
    heard_prompt = modality == "typed" or (
        len(inputs) == 4
        and all(
            isinstance(item.get("input_wer"), (int, float))
            and not isinstance(item.get("input_wer"), bool)
            and item["input_wer"] <= MAX_INPUT_WER
            for item in inputs
        )
    )
    result = {
        "scenario_identity": scenario_identity and scenario_turn_identity,
        "runtime_fixture_identity": runtime_fixture_identity,
        "response_cardinality": len(responses) == 4,
        "input_lifecycle": input_lifecycle,
        "tool_calls": len(responses) == 4
        and len(outputs) == 4
        and all(
            response.get("tool_calls")
            == [{"name": (scenario.get("tool") or {}).get("name"), "arguments": {}}]
            and response.get("tool_call_correlated") is True
            and response.get("function_output_applied") is True
            and response.get("expected_response_any") == [tool_output.get("expected_response")]
            and (response.get("expected_response_none") or [])
            == (tool_output.get("excluded_responses") or [])
            and response.get("expected_text_semantic_required")
            is tool_output.get("semantic_required")
            and response.get("function_output_sent")
            == json.dumps(
                tool_output.get("function_output"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for response, tool_output in zip(responses, outputs, strict=True)
        ),
        "response_correlation": response_correlation,
        "response_brackets": len(responses) == 4
        and all(response.get("response_brackets_balanced") is True for response in responses),
        "eotr_valid": len(checks) == 4 and all(check.get("eotr_valid") is True for check in checks),
        "function_channel_idle": len(checks) == 4
        and all(check.get("function_channel_idle") is True for check in checks),
        "post_eotr_audible": len(checks) == 4
        and all(check.get("post_eotr_audible") is True for check in checks),
        "bounded_watchdog_close": len(checks) == 4
        and all(check.get("bounded_watchdog_close") is True for check in checks),
        "text_nonempty": len(checks) == 4
        and all(check.get("text_nonempty") is True for check in checks),
        "audio_nonempty": len(checks) == 4
        and all(check.get("audio_nonempty") is True for check in checks),
        "cycle_cardinality": cardinality.get("passed") is True,
        "protocol_errors_absent": not errors,
        "clean_session_close": isinstance(session_closed, dict)
        and session_closed.get("reason") == "client_stop"
        and session_closed.get("status") == "completed",
        "heard_prompt": heard_prompt,
    }
    if tuple(result) != structural_conjuncts_for_scenario(scenario):
        raise RuntimeError("tool freshness structural conjunct inventory drifted")
    return result
