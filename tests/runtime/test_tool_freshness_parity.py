from __future__ import annotations

import base64
import importlib.util
import json
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest


def load_module():
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        path = Path("tools/qualification/tool_freshness_parity.py")
        spec = importlib.util.spec_from_file_location("tool_freshness_parity", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def structural_fixture(module, modality="typed"):
    scenario = module.scenario_document()
    fixtures = [
        {
            "case_id": f"tool-freshness-{ordinal}",
            "ordinal": ordinal,
            "text": scenario["prompts"][ordinal - 1],
        }
        for ordinal in range(1, 5)
    ]
    inputs = []
    responses = []
    checks = []
    for ordinal in range(1, 5):
        turn_id = f"turn-{ordinal}"
        input_item = {
            "ordinal": ordinal,
            "case_id": f"tool-freshness-{ordinal}",
            "input_modality": modality,
            "turn_id": turn_id,
            "expected_input_text": scenario["prompts"][ordinal - 1],
            "requested_text": (scenario["prompts"][ordinal - 1] if modality == "typed" else None),
            "accepted": True,
            "started": True,
            "disposition": "completed",
            "speech_started": True,
            "transcription_source": "typed" if modality == "typed" else "microphone",
            "transcription_job_id": f"job-{ordinal}" if modality == "typed" else None,
            "turn_started_ack": True,
            "committed_ack": True,
            "client_turn_id": ordinal,
            "turn_started_turn_id": turn_id,
            "committed_turn_id": turn_id,
            "transcription_turn_id": turn_id,
            "displayed_input_text": scenario["prompts"][ordinal - 1],
            "job_id": f"job-{ordinal}" if modality == "typed" else None,
            "input_wer": 0.1 if modality == "speech" else None,
            "event_counts": (
                {
                    "input_text.accepted": 1,
                    "input_text.injection_started": 1,
                    "input_text.injection_finished": 1,
                    "input_audio_buffer.speech_started": 1,
                    "conversation.item.input_audio_transcription.completed": 1,
                }
                if modality == "typed"
                else {
                    "input_audio_buffer.speech_started": 1,
                    "input_audio_buffer.turn_started": 1,
                    "input_audio_buffer.committed": 1,
                    "conversation.item.input_audio_transcription.completed": 1,
                }
            ),
            "event_correlations": [True] * (5 if modality == "typed" else 4),
        }
        inputs.append(input_item)
        responses.append(
            {
                "turn_id": turn_id,
                "case_id": f"tool-freshness-{ordinal}",
                "ordinal": ordinal,
                "response_id": f"response-{ordinal}",
                "tool_calls": [{"name": scenario["tool"]["name"], "arguments": {}}],
                "tool_call_correlated": True,
                "function_output_applied": True,
                "response_brackets_balanced": True,
                "protocol_correlation": True,
                "expected_response_any": [
                    scenario["tool_outputs"][ordinal - 1]["expected_response"]
                ],
                "expected_response_none": scenario["tool_outputs"][ordinal - 1][
                    "excluded_responses"
                ],
                "expected_text_semantic_required": True,
                "function_output_sent": json.dumps(
                    scenario["tool_outputs"][ordinal - 1]["function_output"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "event_counts": {kind: 1 for kind in module.RESPONSE_SINGLETON_EVENT_TYPES},
                "event_correlations": [True] * 6,
            }
        )
        checks.append(
            {
                "eotr_valid": True,
                "function_channel_idle": True,
                "post_eotr_audible": True,
                "bounded_watchdog_close": True,
                "text_nonempty": True,
                "audio_nonempty": True,
                "text_semantic_match": False,
            }
        )
    return {
        "modality": modality,
        "inputs": inputs,
        "responses": responses,
        "checks": checks,
        "cardinality": {"passed": True},
        "errors": [],
        "session_closed": {"reason": "client_stop", "status": "completed"},
        "scenario": scenario,
        "fixtures": fixtures,
        "scenario_identity": True,
        "runtime_fixture_identity": True,
    }


def test_structural_gate_enumerates_exact_conjuncts_and_excludes_semantics() -> None:
    module = load_module()
    from nemotron_voicechat_runtime import tool_freshness_contract

    assert module.derive_structural_conjuncts is (
        tool_freshness_contract.derive_structural_conjuncts
    )
    assert module.evaluate_tool_response_channels is (
        tool_freshness_contract.evaluate_tool_response_channels
    )
    arguments = structural_fixture(module)

    result = module.derive_structural_conjuncts(**arguments)

    assert tuple(result) == module.STRUCTURAL_CONJUNCTS
    assert "text_semantic_match" not in result
    assert all(result.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_runtime_marks_forward_policy_and_reverse_nonpromotion(
    tmp_path: Path, monkeypatch, reverse: bool
) -> None:
    module = load_module()
    runtime_id = f"sha256:{'a' * 64}"
    scenario_sha = module.canonical_sha256(module.scenario_document())
    manifest = {
        "sha256": "b" * 64,
        "scenario_sha256": scenario_sha,
        "provenance": {"runtime_image_id": runtime_id},
    }
    monkeypatch.setattr(
        module,
        "validate_tool_freshness_fixtures",
        lambda *_a, **_k: (manifest, (b"pcm",) * 4),
    )
    monkeypatch.setattr(module, "validate_fixture_source_provenance", lambda *_a, **_k: None)

    class Health:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_a, **_k: Health(),
    )
    monkeypatch.setattr(
        module.json,
        "load",
        lambda _response: {"active_client": False, "typed_input": {"ready": True}},
    )
    observed: list[str] = []

    async def fake_run_arm(_args, *, modality, **_kwargs):
        observed.append(modality)
        arm = _args.output / modality
        arm.mkdir(parents=True)
        (arm / "report.json").write_text("{}", encoding="utf-8")
        return {
            "session_id": f"session-{modality}",
            "runtime_provenance": {"runtime_image_id": runtime_id},
            "experiment_sha256": module.experiment_sha256(
                scenario_sha256=scenario_sha,
                fixture_manifest_sha256=manifest["sha256"],
            ),
            "structural_passed": True,
        }

    monkeypatch.setattr(module, "run_arm", fake_run_arm)
    args = Namespace(
        output=tmp_path / "runtime",
        fixture_manifest=tmp_path / "fixtures/manifest.json",
        runtime_image_id=runtime_id,
        health_url="http://health",
        nonpromotable_reverse_order=reverse,
    )

    result = await module.run(args)

    expected_order = ["speech", "typed"] if reverse else ["typed", "speech"]
    assert observed == expected_order
    assert result["kind"] == (
        "tool_freshness_reverse_order_diagnostic_runtime"
        if reverse
        else "tool_freshness_parity_runtime"
    )
    assert result["execution_order"] == expected_order
    assert result["session_ids"] == [f"session-{item}" for item in expected_order]
    if reverse:
        assert result["promotion_eligible"] is False
        assert "comparison_policy" not in result
    else:
        assert "promotion_eligible" not in result
        assert result["comparison_policy"] == "parity_margin_v2"
    assert result["passed"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("scenario_identity", False),
        lambda value: value.__setitem__("runtime_fixture_identity", False),
        lambda value: value["responses"].pop(),
        lambda value: value["inputs"][0].__setitem__("disposition", "error"),
        lambda value: value["responses"][0].__setitem__("tool_calls", []),
        lambda value: value["responses"][0].__setitem__("tool_call_correlated", False),
        lambda value: value["responses"][0].__setitem__("expected_response_any", ["wrong-code"]),
        lambda value: value["responses"][0].__setitem__("function_output_sent", "{}"),
        lambda value: value["responses"][0].__setitem__("turn_id", "wrong"),
        lambda value: value["responses"][0].__setitem__("response_brackets_balanced", False),
        lambda value: value["checks"][0].__setitem__("eotr_valid", False),
        lambda value: value["checks"][0].__setitem__("function_channel_idle", False),
        lambda value: value["checks"][0].__setitem__("post_eotr_audible", False),
        lambda value: value["checks"][0].__setitem__("bounded_watchdog_close", False),
        lambda value: value["checks"][0].__setitem__("text_nonempty", False),
        lambda value: value["checks"][0].__setitem__("audio_nonempty", False),
        lambda value: value["cardinality"].__setitem__("passed", False),
        lambda value: value["errors"].append({"type": "error"}),
        lambda value: value.__setitem__("session_closed", {"reason": "fatal"}),
        lambda value: value["inputs"][0]["event_counts"].update({"input_text.accepted": 2}),
        lambda value: value["responses"][0]["event_correlations"].append(False),
    ],
)
def test_each_absolute_structural_family_can_turn_the_gate_red(mutation) -> None:
    module = load_module()
    arguments = structural_fixture(module)
    mutation(arguments)

    result = module.derive_structural_conjuncts(**arguments)

    assert not all(result.values())


def test_turns_are_bound_to_ordered_scenario_fixture_and_unique_ids() -> None:
    module = load_module()
    arguments = structural_fixture(module)

    changed_prompt = deepcopy(arguments)
    changed_prompt["inputs"][0]["expected_input_text"] = "arbitrary"
    changed_prompt["inputs"][0]["requested_text"] = "arbitrary"
    assert not all(module.derive_structural_conjuncts(**changed_prompt).values())

    changed_case = deepcopy(arguments)
    changed_case["inputs"][0]["case_id"] = "wrong-case"
    changed_case["responses"][0]["case_id"] = "wrong-case"
    assert not all(module.derive_structural_conjuncts(**changed_case).values())

    duplicate_turn = deepcopy(arguments)
    duplicate_turn["inputs"][1]["turn_id"] = duplicate_turn["inputs"][0]["turn_id"]
    duplicate_turn["inputs"][1]["transcription_turn_id"] = duplicate_turn["inputs"][0]["turn_id"]
    duplicate_turn["responses"][1]["turn_id"] = duplicate_turn["inputs"][0]["turn_id"]
    assert not all(module.derive_structural_conjuncts(**duplicate_turn).values())


def test_speech_heard_gate_requires_correlated_acks_and_bounded_wer() -> None:
    module = load_module()
    arguments = structural_fixture(module, modality="speech")
    assert all(module.derive_structural_conjuncts(**arguments).values())

    for field, value in (
        ("turn_started_ack", False),
        ("committed_ack", False),
        ("turn_started_turn_id", "wrong"),
        ("committed_turn_id", "wrong"),
        ("transcription_turn_id", "wrong"),
        ("transcription_source", "typed"),
        ("displayed_input_text", ""),
        ("input_wer", 0.3501),
    ):
        mutated = deepcopy(arguments)
        mutated["inputs"][0][field] = value
        result = module.derive_structural_conjuncts(**mutated)
        assert not all(result.values())


def test_fixture_synthesis_must_exactly_match_live_typed_worker() -> None:
    module = load_module()
    synthesis = {
        "language": "english_2026-04",
        "voice": "alba",
        "threads": 4,
        "cpus": [7, 8, 9, 15],
        "base_seed": 0,
        "seed_scheme": "sha256-text-plus-base-v1",
        "package_version": "2.1.0",
        "assets": {"a": {"bytes": 1, "sha256": "a" * 64}},
    }
    health = {
        "ready": True,
        "pid": 10,
        "socket": "/tmp/pocket.sock",
        **{key: value for key, value in synthesis.items() if key != "base_seed"},
        "seed": 0,
    }

    assert module.fixture_synthesis_identity({"synthesis": synthesis}) == (
        module.health_synthesis_identity(health)
    )
    health["threads"] = 3
    assert module.fixture_synthesis_identity({"synthesis": synthesis}) != (
        module.health_synthesis_identity(health)
    )


@pytest.mark.parametrize("modality", ["typed", "speech"])
@pytest.mark.asyncio
async def test_run_arm_parses_exact_correlated_wire_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, modality: str
) -> None:
    module = load_module()
    from generate_tool_freshness_fixtures import scenario_document

    from nemotron_voicechat_runtime.provenance import RUNTIME_PROVENANCE_REQUIRED_SOURCES

    scenario = scenario_document()
    scenario_sha256 = module.canonical_sha256(scenario)
    runtime_image_id = f"sha256:{'a' * 64}"
    runtime_provenance = {
        "runtime_image_id": runtime_image_id,
        "source_sha256": {name: "b" * 64 for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES},
    }
    synthesis = {
        "language": "english_2026-04",
        "voice": "alba",
        "quantize": True,
        "threads": 4,
        "cpus": [7, 8, 9, 15],
        "base_seed": 0,
        "seed_scheme": "sha256-text-plus-base-v1",
        "package_version": "2.1.0",
        "assets": {"asset": {"bytes": 1, "sha256": "c" * 64}},
    }
    from nemotron_voicechat_runtime.tool_freshness_fixture import (
        FIXTURE_KIND,
        FIXTURE_SCHEMA,
    )

    manifest = {
        "schema": FIXTURE_SCHEMA,
        "kind": FIXTURE_KIND,
        "scenario_sha256": scenario_sha256,
        "fixtures": [
            {
                "case_id": f"tool-freshness-{ordinal}",
                "ordinal": ordinal,
                "text": prompt,
            }
            for ordinal, prompt in enumerate(scenario["prompts"], 1)
        ],
        "synthesis": synthesis,
        "provenance": {
            "runtime_image": "runtime:test",
            "runtime_image_id": runtime_image_id,
            "source_sha256": {"fixture": "d" * 64},
        },
    }
    from nemotron_voicechat_runtime.tool_freshness_fixture import self_hash

    manifest["sha256"] = self_hash(manifest)
    health = {
        "active_client": False,
        "typed_input": {
            "ready": True,
            "language": synthesis["language"],
            "voice": synthesis["voice"],
            "threads": synthesis["threads"],
            "cpus": synthesis["cpus"],
            "seed": synthesis["base_seed"],
            "seed_scheme": synthesis["seed_scheme"],
            "package_version": synthesis["package_version"],
            "assets": synthesis["assets"],
        },
        "checkpoint": {"runtime_provenance": runtime_provenance},
    }
    session_id = f"session-{modality}"
    events = [
        {
            "type": "session.created",
            "protocol": {"name": "voicechat.realtime", "version": 3},
            "capabilities": {
                "input_turn_detection": "client_smart_turn_v1",
                "typed_input": True,
            },
            "session": {
                "id": session_id,
                "checkpoint": {"runtime_provenance": runtime_provenance},
            },
        },
        {
            "type": "session.updated",
            "session": {
                "id": session_id,
                "instructions": scenario["instructions"],
                "tools": [scenario["tool"]],
            },
        },
    ]
    for ordinal, tool_output in enumerate(scenario["tool_outputs"], 1):
        case_id = f"tool-freshness-{ordinal}"
        turn_id = f"turn-{ordinal}"
        response_id = f"response-{ordinal}"
        if modality == "typed":
            # job_id has a random suffix; the fake socket fills it from the request.
            events.extend(
                [
                    {"type": "input_text.accepted", "job_id": "$JOB"},
                    {"type": "input_text.injection_started", "job_id": "$JOB"},
                    {
                        "type": "input_audio_buffer.speech_started",
                        "source": "typed",
                        "job_id": "$JOB",
                        "turn_id": turn_id,
                    },
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "source": "typed",
                        "job_id": "$JOB",
                        "turn_id": turn_id,
                        "transcript": scenario["prompts"][ordinal - 1],
                    },
                    {
                        "type": "input_text.injection_finished",
                        "job_id": "$JOB",
                        "disposition": "completed",
                    },
                ]
            )
        else:
            events.extend(
                [
                    {
                        "type": "input_audio_buffer.speech_started",
                        "source": "microphone",
                        "turn_id": turn_id,
                    },
                    {
                        "type": "input_audio_buffer.turn_started",
                        "client_turn_id": ordinal,
                        "client_event_id": f"{case_id}-start",
                        "turn_id": turn_id,
                    },
                    {
                        "type": "input_audio_buffer.committed",
                        "client_turn_id": ordinal,
                        "client_event_id": f"{case_id}-commit",
                        "turn_id": turn_id,
                    },
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "source": "microphone",
                        "turn_id": turn_id,
                        "transcript": scenario["prompts"][ordinal - 1],
                    },
                ]
            )
        events.extend(
            [
                {
                    "type": "response.created",
                    "turn_id": turn_id,
                    "response_id": response_id,
                },
                {
                    "type": "response.function_call_arguments.done",
                    "turn_id": turn_id,
                    "response_id": response_id,
                    "call_id": f"call-{ordinal}",
                    "name": scenario["tool"]["name"],
                    "arguments": "{}",
                },
                {
                    "type": "conversation.item.function_call_output.applied",
                    "call_id": f"call-{ordinal}",
                },
                {
                    "type": "voicechat.metrics",
                    "turn_id": turn_id,
                    "response_id": response_id,
                    "transport_frame": ordinal * 10,
                    "turn_state": {
                        "function_calling": {
                            "eotr_observed": True,
                            "eotr_feedback_committed": True,
                            "eotr_observed_count": ordinal,
                            "eotr_observed_frame": ordinal * 100,
                            "eotr_wait_steps": 1,
                            "eotr_timeout": False,
                            "eotr_token_id": 5,
                            "eotr_first_post_drain_token_id": 5,
                            "eotr_first_post_drain_token_frame": ordinal * 100,
                            "eotr_unexpected_token_id": None,
                            "eotr_unexpected_token_frame": None,
                            "eotr_feedback_token_id": 5,
                            "eotr_feedback_frame": ordinal * 100,
                            "client_eou_sotc_committed_count": ordinal,
                            "post_fc_client_bos_forced_count": ordinal,
                        }
                    },
                },
                {
                    "type": "voicechat.metrics",
                    "turn_id": turn_id,
                    "response_id": response_id,
                    "transport_frame": ordinal * 10 + 1,
                    "output_audio": {"rms_dbfs": -20.0},
                    "turn_state": {
                        "function_calling": {
                            "effective_token_is_pad": True,
                            "client_eou_sotc_committed_count": ordinal,
                            "post_fc_client_bos_forced_count": ordinal,
                            "eotr_observed_count": ordinal,
                        },
                        "agent_silence_watchdog": {"threshold_dbfs": -90.0},
                    },
                },
                {
                    "type": "response.output_text.delta",
                    "turn_id": turn_id,
                    "response_id": response_id,
                    "delta": tool_output["expected_response"],
                },
                {
                    "type": "response.output_audio.delta",
                    "turn_id": turn_id,
                    "response_id": response_id,
                    "delta": base64.b64encode(b"\x01\x00" * 32).decode(),
                },
                {
                    "type": "response.output_text.done",
                    "turn_id": turn_id,
                    "response_id": response_id,
                },
                {
                    "type": "response.output_audio.done",
                    "turn_id": turn_id,
                    "response_id": response_id,
                },
                {
                    "type": "response.done",
                    "turn_id": turn_id,
                    "response_id": response_id,
                    "status": "completed",
                    "reason": "decoded_silence_watchdog",
                },
            ]
        )
    events.append({"type": "session.closed", "status": "completed", "reason": "client_stop"})

    class FakeWebSocket:
        def __init__(self) -> None:
            self.events = events
            self.sent: list[dict] = []
            self.current_job: str | None = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, raw: str) -> None:
            message = json.loads(raw)
            self.sent.append(message)
            if message.get("type") == "input_text.request":
                self.current_job = message["job_id"]

        async def recv(self) -> str:
            event = self.events.pop(0)
            if event.get("job_id") == "$JOB":
                event = {**event, "job_id": self.current_job}
            return json.dumps(event)

    websocket = FakeWebSocket()
    monkeypatch.setattr(module.websockets, "connect", lambda *_args, **_kwargs: websocket)
    sent_audio: list[tuple] = []

    async def fake_send_audio(*args):
        sent_audio.append(args[1:])

    async def fake_silence(_websocket, _case_id, _stop):
        return None

    monkeypatch.setattr(module, "send_audio", fake_send_audio)
    monkeypatch.setattr(module, "send_silence_until_stopped", fake_silence)
    args = Namespace(
        output=tmp_path,
        url="ws://test",
        runtime_image_id=runtime_image_id,
        fixture_manifest=tmp_path / "manifest.json",
        turn_timeout_seconds=10,
        settle_seconds=0,
    )

    report = await module.run_arm(
        args,
        modality=modality,
        scenario=scenario,
        scenario_sha256=scenario_sha256,
        manifest=manifest,
        fixture_pcm=(b"\x00\x00" * 128,) * 4,
        health_snapshot=health,
    )

    assert report["structural_passed"] is True
    assert report["session_id"] == session_id
    assert len([item for item in websocket.sent if item["type"] == "conversation.item.create"]) == 4
    assert len(sent_audio) == (4 if modality == "speech" else 0)
