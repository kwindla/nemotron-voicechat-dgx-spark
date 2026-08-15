from __future__ import annotations

import asyncio
import json
import math
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from nemotron_voicechat_runtime.server import (
    FC_ALWAYS_ACKNOWLEDGE_TOOLS_ENV,
    FC_FAST_TOOL_GRACE_MS_ENV,
    _validated_fc_async_benchmark_phase,
    _validated_tool_wait_evidence,
    fc_always_acknowledge_tools,
    fc_fast_tool_grace_ms,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = runpy.run_path(
    str(REPO_ROOT / "tools/benchmark/tool_call_latency_benchmark.py")
)
ANALYZER = runpy.run_path(str(REPO_ROOT / "tools/benchmark/analyze_tool_call_latency.py"))
FIRST_CHAIN_TOOL = BENCHMARK["FIRST_CHAIN_TOOL"]
SECOND_CHAIN_TOOL = BENCHMARK["SECOND_CHAIN_TOOL"]
SAFETY_SCENARIOS = BENCHMARK["SAFETY_SCENARIOS"]
safety_cell_passed = BENCHMARK["safety_cell_passed"]
evaluate_tool_wait_gate = BENCHMARK["evaluate_tool_wait_gate"]
nearest_rank_percentile = BENCHMARK["nearest_rank_percentile"]
function_output_item = BENCHMARK["function_output_item"]
expected_injection_tokens = BENCHMARK["expected_injection_tokens"]
tool_response_position_contract = BENCHMARK["tool_response_position_contract"]
cell_contract_passed = ANALYZER["cell_contract_passed"]
injection_contract = ANALYZER["injection_contract"]
send_delayed_tool_output = BENCHMARK["send_delayed_tool_output"]
send_speech_carrier_with_continuation_clock = BENCHMARK[
    "send_speech_carrier_with_continuation_clock"
]


def nano_engine_record() -> dict:
    return {
        "schema": 1,
        "request_id": "stream-zero",
        "backend_request_id": "stream-zero",
        "backend_generation": 0,
        "position": 42,
        "action": "append",
        "input_copy_ms": 0.2,
        "submit_ms": 0.3,
        "output_wait_ms": 60.0,
        "parse_ms": 0.1,
        "named_sum_ms": 60.6,
        "total_ms": 60.7,
        "residual_ms": 0.1,
    }


def position_record() -> dict:
    return {
        "schema": 1,
        "async_step": 0,
        "nano_position": 42,
        "monotonic_s": 100.0,
        "embed_ms": 0.1,
        "perception_poll_ms": 0.1,
        "fusion_ms": 0.1,
        "nano_interface_ms": 60.7,
        "nano_engine": nano_engine_record(),
        "eartts_infer_ms": 12.0,
        "codec_decode_ms": 6.0,
        "codec_cpu_copy_ms": 2.0,
        "rnnt_ms": 0.0,
        "named_sum_ms": 81.0,
        "total_ms": 81.1,
        "residual_ms": 0.1,
    }


def phase_record() -> dict:
    stages = {
        "embed_ms": 0.1,
        "perception_poll_ms": 0.1,
        "fusion_ms": 0.1,
        "nano_interface_ms": 60.7,
        "eartts_infer_ms": 12.0,
        "codec_decode_ms": 6.0,
        "codec_cpu_copy_ms": 2.0,
        "rnnt_ms": 0.0,
        "perception_join_ms": 0.2,
    }
    named_sum = sum(stages.values())
    return {
        "schema": 1,
        "cycle_sotc_frame": 41,
        "phase": "tool_call",
        "invocation": 1,
        "request_id": "stream-zero",
        "phase_start_monotonic_s": 100.0,
        "phase_end_monotonic_s": 100.082,
        "wall_ms": named_sum + 0.8,
        "positions_count": 1,
        "positions_retained": 1,
        "positions_truncated": False,
        "nano_position_start": 42,
        "nano_position_end": 42,
        "real_audio_frames": 0,
        "perception_calls": 0,
        "rnnt_steps": 0,
        "tts_calls": 1,
        "eotr_wait_steps": 0,
        "stage_totals_ms": stages,
        "named_sum_ms": named_sum,
        "residual_ms": 0.8,
        "positions": [position_record()],
    }


def tool_wait_record(*, policy: str = "disabled", reminder: bool = True) -> dict:
    grace_ms = 100.0 if policy != "disabled" else 0.0
    fast_path = policy == "fast_grace" and not reminder
    complete = 100.05 if fast_path else 100.5
    phase2 = 100.06 if fast_path else 102.1
    return {
        "schema": 1,
        "tool_name": "get_benchmark_word",
        "policy": policy,
        "grace_ms": grace_ms,
        "tool_started_monotonic_s": 100.0,
        "tool_completed_monotonic_s": complete,
        "grace_expired_monotonic_s": (
            100.1 if policy == "fast_grace" and not fast_path else None
        ),
        "tool_completed_before_reminder": bool(reminder and complete <= 100.11),
        "reminder_text": "I am checking that now." if reminder else None,
        "reminder_variant_sha256": "a" * 64 if reminder else None,
        "reminder_tokens": 8,
        "reminder_started_monotonic_s": 100.11 if reminder else None,
        "reminder_ended_monotonic_s": 102.0 if reminder else None,
        "reminder_samples": 32000 if reminder else 0,
        "phase2_started_monotonic_s": phase2,
        "result_available_to_phase2_ms": (phase2 - complete) * 1000.0,
        "fast_path": fast_path,
    }


def test_tool_wait_evidence_validates_disabled_fast_and_always_acknowledge() -> None:
    always_without_grace = tool_wait_record(policy="always_acknowledge", reminder=True)
    always_without_grace["grace_ms"] = 0.0
    for record in (
        tool_wait_record(),
        tool_wait_record(policy="fast_grace", reminder=False),
        tool_wait_record(policy="fast_grace", reminder=True),
        tool_wait_record(policy="always_acknowledge", reminder=True),
        always_without_grace,
    ):
        validated = _validated_tool_wait_evidence(record)
        assert validated == record
        assert validated is not record


def test_tool_wait_evidence_rejects_incoherent_or_mutated_records() -> None:
    mutations = (
        lambda value: value.update(extra=True),
        lambda value: value.update(schema=2),
        lambda value: value.update(policy="unknown"),
        lambda value: value.update(tool_started_monotonic_s=101.0),
        lambda value: value.update(tool_completed_monotonic_s=103.0),
        lambda value: value.update(result_available_to_phase2_ms=1.0),
        lambda value: value.update(tool_completed_before_reminder=True),
    )
    for mutate in mutations:
        candidate = tool_wait_record(policy="fast_grace", reminder=False)
        mutate(candidate)
        assert _validated_tool_wait_evidence(candidate) is None

    for mutate in (
        lambda value: value.update(reminder_variant_sha256="not-a-digest"),
        lambda value: value.update(reminder_samples=0),
        lambda value: value.update(reminder_ended_monotonic_s=None),
    ):
        candidate = tool_wait_record(policy="fast_grace", reminder=True)
        mutate(candidate)
        assert _validated_tool_wait_evidence(candidate) is None

    candidate = tool_wait_record(policy="fast_grace", reminder=True)
    candidate["fast_path"] = True
    assert _validated_tool_wait_evidence(candidate) is None
    candidate = tool_wait_record(policy="fast_grace", reminder=True)
    candidate["grace_expired_monotonic_s"] = 100.12
    assert _validated_tool_wait_evidence(candidate) is None
    candidate = tool_wait_record()
    candidate["grace_ms"] = 100.0
    assert _validated_tool_wait_evidence(candidate) is None


@pytest.mark.parametrize("value", ["true", "nan", "inf", "-1", "1000.1"])
def test_fast_tool_grace_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(FC_FAST_TOOL_GRACE_MS_ENV, value)
    with pytest.raises(SystemExit):
        fc_fast_tool_grace_ms()


def test_fast_tool_policy_environment_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FC_FAST_TOOL_GRACE_MS_ENV, "100")
    monkeypatch.setenv(FC_ALWAYS_ACKNOWLEDGE_TOOLS_ENV, "slow_tool, audited_tool")
    assert fc_fast_tool_grace_ms() == 100.0
    assert fc_always_acknowledge_tools() == ["slow_tool", "audited_tool"]
    monkeypatch.setenv(FC_ALWAYS_ACKNOWLEDGE_TOOLS_ENV, "slow_tool,slow_tool")
    with pytest.raises(SystemExit):
        fc_always_acknowledge_tools()


def test_reminder_gate_rederives_transport_and_fast_path_outcomes() -> None:
    slow = {
        "tool_wait_evidence": tool_wait_record(),
        "reminder_transport_witness": {
            "generated_samples": 32000,
            "received_samples": 32000,
            "audible_chunks": 4,
        },
    }
    assert evaluate_tool_wait_gate(
        slow, expected_policy="disabled", expect_reminder=True
    )["passed"]
    fast = {
        "tool_wait_evidence": tool_wait_record(policy="fast_grace", reminder=False),
        "reminder_transport_witness": None,
    }
    assert evaluate_tool_wait_gate(
        fast, expected_policy="fast_grace", expect_reminder=False
    )["passed"]
    fast["tool_wait_evidence"]["result_available_to_phase2_ms"] = math.nan
    assert not evaluate_tool_wait_gate(
        fast, expected_policy="fast_grace", expect_reminder=False
    )["passed"]


def test_reminder_rendering_can_be_gated_without_enshrining_transport_defect() -> None:
    cell = {
        "tool_wait_evidence": tool_wait_record(policy="fast_grace", reminder=True),
        "reminder_transport_witness": {
            "generated_samples": 32000,
            "received_samples": 0,
            "audible_chunks": 0,
        },
    }
    assert not evaluate_tool_wait_gate(
        cell,
        expected_policy="fast_grace",
        expect_reminder=True,
        require_reminder_transport=True,
    )["passed"]
    gate = evaluate_tool_wait_gate(
        cell,
        expected_policy="fast_grace",
        expect_reminder=True,
        require_reminder_transport=False,
    )
    assert gate["passed"]
    assert gate["reminder_rendered"] is True
    assert gate["reminder_transported"] is False


def test_nearest_rank_p95_uses_the_tenth_value_for_ten_replicates() -> None:
    assert nearest_rank_percentile([float(value) for value in range(1, 11)], 0.95) == 10.0
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 0.95)


def test_delayed_tool_output_runs_without_blocking_receive_progress() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    async def scenario() -> tuple[int, list[dict]]:
        websocket = FakeWebSocket()
        task = asyncio.create_task(
            send_delayed_tool_output(
                websocket,
                ordinal=7,
                call_id="call-seven",
                payload={
                    "output": '{"fresh_word":"River"}',
                    "model_output": "River is fresh.",
                },
                use_model_output=False,
                delay_ms=20.0,
            )
        )
        receive_progress = 0
        while not task.done():
            receive_progress += 1
            await asyncio.sleep(0)
        await task
        return receive_progress, websocket.sent

    receive_progress, sent = asyncio.run(scenario())
    assert receive_progress > 1
    assert sent == [
        {
            "type": "conversation.item.create",
            "event_id": "benchmark-output-7",
            "item": {
                "type": "function_call_output",
                "call_id": "call-seven",
                "output": '{"fresh_word":"River"}',
            },
        }
    ]


def test_model_output_payload_is_explicit_and_changes_expected_injection_size() -> None:
    payload = {
        "output": "full structured result",
        "realized_tokens": 72,
        "model_output": "River is fresh.",
        "model_output_tokens": 19,
    }
    assert function_output_item(
        call_id="call-one", payload=payload, use_model_output=True
    ) == {
        "type": "function_call_output",
        "call_id": "call-one",
        "output": "full structured result",
        "model_output": "River is fresh.",
    }
    assert expected_injection_tokens(payload, False) == 72
    assert expected_injection_tokens(payload, True) == 19


def test_tool_response_position_contract_fails_closed_without_exact_phase_count() -> None:
    payload = {"realized_tokens": 72, "model_output_tokens": 20}
    phases = [{"phase": "tool_response", "positions_count": 21}]
    assert tool_response_position_contract(phases, payload, True) == (21, True)
    assert tool_response_position_contract([], payload, True) == (21, False)
    phases[0]["positions_count"] = 20
    assert tool_response_position_contract(phases, payload, True) == (21, False)


def test_model_output_injection_contract_uses_phase_positions_not_missing_ack() -> None:
    cell = {
        "ordinal": 1,
        "block": 1,
        "call_id": "call-one",
        "errors": [],
        "payload": {
            "verification_word": "River",
            "realized_tokens": 72,
            "model_output_tokens": 20,
        },
        "model_output_enabled": True,
        "function_output_tokens_ack": None,
        "text": "The fresh word is River.",
        "first_audible_monotonic_s": 101.0,
        "phases": [
            {
                "phase": "tool_call",
                "positions_count": 27,
                "positions_truncated": False,
            },
            {
                "phase": "tool_response",
                "positions_count": 21,
                "positions_truncated": False,
            },
        ],
    }
    assert injection_contract(cell) == {
        "ordinal": 1,
        "full_output_tokens": 72,
        "injection_source": "model_output",
        "injected_tokens": 20,
        "expected_positions": 21,
        "actual_positions": 21,
        "function_output_tokens_ack": None,
        "acknowledgement_matches": True,
        "passed": True,
    }
    assert cell_contract_passed(cell) is True

    wrong_positions = deepcopy(cell)
    wrong_positions["phases"][1]["positions_count"] = 22
    assert injection_contract(wrong_positions)["passed"] is False
    assert cell_contract_passed(wrong_positions) is False

    wrong_ack = deepcopy(cell)
    wrong_ack["function_output_tokens_ack"] = 72
    assert injection_contract(wrong_ack)["passed"] is False
    assert cell_contract_passed(wrong_ack) is False


def test_fc_async_benchmark_phase_validates_and_copies() -> None:
    raw = phase_record()
    validated = _validated_fc_async_benchmark_phase(raw)
    assert validated == raw
    assert validated is not raw
    assert validated["positions"] is not raw["positions"]


def test_fc_async_benchmark_phase_rejects_mutated_or_incomplete_evidence() -> None:
    for mutate in (
        lambda value: value.update(extra=True),
        lambda value: value.update(phase="tool_response"),
        lambda value: value.update(positions_count=2),
        lambda value: value.update(named_sum_ms=0.0),
        lambda value: value["stage_totals_ms"].update(nano_interface_ms=-1.0),
        lambda value: value["positions"][0].update(nano_engine=None),
        lambda value: value["positions"][0]["nano_engine"].update(action="unknown"),
        lambda value: value["positions"][0]["nano_engine"].update(total_ms=1.0),
    ):
        candidate = deepcopy(phase_record())
        mutate(candidate)
        assert _validated_fc_async_benchmark_phase(candidate) is None


def test_diagnostic_vllm_installer_binds_the_qualified_core_and_stays_opt_in() -> None:
    index = json.loads((REPO_ROOT / "tools/provenance/qualified-deltas/index.json").read_text())
    core = next(item for item in index["files"] if item["path"] == "v1/engine/core.py")
    installer = (REPO_ROOT / "tools/benchmark/install_vllm_tool_latency_ledger.py").read_text()
    assert core["sha256"] in installer
    assert "if not self._voicechat_benchmark_enabled:" in installer
    assert "time.perf_counter()" in installer
    assert "torch.cuda.synchronize" not in installer
    assert "[VOICECHAT_ENGINE_STEP]" in installer
    assert 'logger.warning("[VOICECHAT_ENGINE_STEP]' in installer


def test_speech_benchmark_patch_is_opt_in_and_has_no_new_cuda_sync() -> None:
    patch = (
        REPO_ROOT
        / "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch"
    ).read_text()
    assert 'S2S_FC_ASYNC_BENCHMARK", "0"' in patch
    assert "if not self._fc_async_benchmark_enabled:" in patch
    assert "nano_engine_benchmark" in patch
    assert "drain_fc_async_benchmark" in patch
    async_method = patch[
        patch.index("     def _run_fc_async_steps(") : patch.index(
            "     def _prepare_system_prompt_embeddings("
        )
    ]
    assert "torch.cuda.synchronize" not in async_method


def test_fast_tool_grace_patch_is_opt_in_waits_on_future_and_retains_policy() -> None:
    patch = (
        REPO_ROOT
        / "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch"
    ).read_text()
    assert 'getattr(s2s_cfg, "fc_fast_tool_grace_ms", 0.0)' in patch
    assert "timeout=fast_tool_grace_ms / 1000.0" in patch
    assert '_policy = (' in patch
    assert '"always_acknowledge"' in patch
    assert 'fc_state["tool_wait_evidence"] = {' in patch
    grace_start = patch.index("if fast_tool_grace_ms > 0.0 and not _always_acknowledge:")
    reminder_start = patch.index("# Layer 1", grace_start)
    assert "time.sleep" not in patch[grace_start:reminder_start]


def test_always_acknowledge_normalizer_accepts_omegaconf_listconfig() -> None:
    class ListConfig:
        """Minimal host-side stand-in for OmegaConf's list-like container."""

        def __init__(self, values: list[str]) -> None:
            self._values = values

        def __iter__(self):
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    patch = (
        REPO_ROOT
        / "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch"
    ).read_text()
    start = patch.index("+def _normalize_fc_always_acknowledge_tools(")
    end = patch.index("\n _CITY_COORDS", start)
    executable = "\n".join(
        line[1:] for line in patch[start:end].splitlines() if line.startswith("+")
    )
    namespace: dict[str, object] = {"ListConfig": ListConfig}
    exec(executable, namespace)  # noqa: S102 - execute the exact canonical patch helper
    normalize = namespace["_normalize_fc_always_acknowledge_tools"]
    configured = ListConfig(["get_benchmark_word"])
    assert isinstance(configured, ListConfig)
    assert normalize(configured) == frozenset({"get_benchmark_word"})
    assert normalize(" first, second ") == frozenset({"first", "second"})
    with pytest.raises(ValueError):
        normalize(ListConfig(["duplicate", "duplicate"]))
    with pytest.raises(TypeError):
        normalize({"not": "a sequence"})


def test_fc_interrupt_reset_preserves_explicit_client_eou_authority() -> None:
    patch = (
        REPO_ROOT
        / "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch"
    ).read_text()
    helper = patch.index("+\tdef _reset_rnnt_after_fc_interrupt(")
    reset = patch.index("+\tdef _terminate_fc_async_for_reset(", helper)
    block = patch[helper:reset]
    assert 'getattr(wrapper, "_external_user_eou_mode", False)' in block
    assert 'getattr(wrapper, "set_external_user_eou_mode", None)' in block
    assert "mode_fn(True)" in block
    assert patch.count("self._reset_rnnt_after_fc_interrupt()") == 2

    executable = "\n".join(
        line[1:] for line in block.splitlines() if line.startswith("+")
    )
    namespace: dict[str, object] = {}
    exec(f"class InterruptHarness:\n{executable}", namespace)  # noqa: S102

    class Wrapper:
        def __init__(self) -> None:
            self._external_user_eou_mode = True

        def _reset_rnnt_turn_taking_state(self) -> None:
            self._external_user_eou_mode = False

        def set_external_user_eou_mode(self, enabled: bool) -> None:
            self._external_user_eou_mode = enabled

        def request_user_eou(self) -> None:
            if not self._external_user_eou_mode:
                raise RuntimeError("external user EOU mode is not enabled")

    wrapper = Wrapper()
    harness = namespace["InterruptHarness"]()
    harness.s2s_model = wrapper
    harness._reset_rnnt_after_fc_interrupt()
    wrapper.request_user_eou()
    assert wrapper._external_user_eou_mode is True

    wrapper._reset_rnnt_turn_taking_state()
    assert wrapper._external_user_eou_mode is False


def test_barge_carrier_continues_model_clock_after_commit() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.events: list[dict] = []

        async def send(self, raw: str) -> None:
            self.events.append(json.loads(raw))

    async def scenario() -> list[dict]:
        websocket = FakeWebSocket()
        stop = asyncio.Event()
        carrier_done = asyncio.Event()
        task = asyncio.create_task(
            send_speech_carrier_with_continuation_clock(
                websocket,
                bytes(640),
                prefix="barge-test",
                client_turn_id=2,
                real_time=False,
                stop=stop,
                carrier_done=carrier_done,
            )
        )
        await asyncio.wait_for(carrier_done.wait(), timeout=1.0)
        for _ in range(100):
            if any(
                event.get("event_id") == "barge-test-continuation-silence-0"
                for event in websocket.events
            ):
                break
            await asyncio.sleep(0)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        return websocket.events

    events = asyncio.run(scenario())
    event_types = [event["type"] for event in events]
    commit_index = event_types.index("input_audio_buffer.commit")
    continuation_index = next(
        index
        for index, event in enumerate(events)
        if event.get("event_id") == "barge-test-continuation-silence-0"
    )
    assert event_types[0] == "input_audio_buffer.turn_start"
    assert event_types[commit_index - 1] == "input_audio_buffer.append"
    assert continuation_index > commit_index
    assert events[commit_index]["client_turn_id"] == 2


def test_diagnostic_image_binds_the_audited_base_by_id_without_network() -> None:
    builder = (REPO_ROOT / "container/build-tool-latency-benchmark.sh").read_text()
    dockerfile = (REPO_ROOT / "container/Dockerfile.tool-latency-benchmark").read_text()
    assert 'runtime_id=$(docker image inspect' in builder
    assert 'docker tag "${runtime_id}" "${runtime_tag}"' in builder
    assert '--build-arg "RUNTIME_IMAGE_ID=${runtime_id}"' in builder
    assert "--network none" in builder
    assert "benchmark-base-image-id" in builder
    assert "benchmark-base-image-id" in dockerfile


def safety_report(scenario: str) -> dict:
    trigger = None
    calls: list[dict] = []
    applied: list[dict] = []
    response_done_count = 1
    text = "The benchmark safety check is ready."
    if scenario == "chained_two_call":
        calls = [
            {"call_id": "first", "name": FIRST_CHAIN_TOOL["name"]},
            {"call_id": "second", "name": SECOND_CHAIN_TOOL["name"]},
        ]
        applied = [{"call_id": "first"}, {"call_id": "second"}]
        text = "River and Meadow are the fresh words."
    elif scenario.endswith("interruption"):
        trigger = {
            "phase": (
                "tool_call" if scenario == "call_emission_interruption" else "tool_response"
            ),
            "reason": (
                "client_send_during_call_emission"
                if scenario == "call_emission_interruption"
                else "client_send_during_output_recovery"
            ),
            "turn_start_sent_monotonic_s": 9.0,
            "sent": True,
            "completed": True,
            "turn_started_ack": True,
            "committed_ack": True,
            "input_transcript_completed": True,
            "turn_id": "barge-turn",
        }
        if scenario == "result_injection_interruption":
            calls = [{"call_id": "result", "name": "get_benchmark_word"}]
            applied = [{"call_id": "result"}]
            response_done_count = 2
    return {
        "scenario": scenario,
        "errors": [],
        "session_closed": True,
        "first_audible_monotonic_s": 10.0,
        "response_done_count": response_done_count,
        "response_done_turn_ids": ["barge-turn"] if trigger is not None else ["typed-turn"],
        "function_calls": calls,
        "function_outputs_applied": applied,
        "barge_trigger": trigger,
        "text": text,
    }


def test_safety_scenarios_are_frozen_and_rederive_green() -> None:
    assert SAFETY_SCENARIOS == (
        "no_tool",
        "chained_two_call",
        "call_emission_interruption",
        "result_injection_interruption",
    )
    assert all(safety_cell_passed(safety_report(scenario)) for scenario in SAFETY_SCENARIOS)


def test_safety_rederivation_rejects_each_structural_failure() -> None:
    mutations = (
        ("no_tool", lambda report: report["function_calls"].append({"call_id": "bad"})),
        ("chained_two_call", lambda report: report["function_calls"].reverse()),
        (
            "call_emission_interruption",
            lambda report: report["barge_trigger"].update(committed_ack=False),
        ),
        (
            "call_emission_interruption",
            lambda report: report["function_calls"].append({"call_id": "published"}),
        ),
        (
            "result_injection_interruption",
            lambda report: report.update(response_done_count=1),
        ),
        (
            "result_injection_interruption",
            lambda report: report["barge_trigger"].update(phase="tool_call"),
        ),
        (
            "result_injection_interruption",
            lambda report: report["barge_trigger"].update(
                reason="server_observed_model_interruption"
            ),
        ),
        (
            "result_injection_interruption",
            lambda report: report.update(response_done_turn_ids=["other-turn"]),
        ),
    )
    for scenario, mutate in mutations:
        report = safety_report(scenario)
        mutate(report)
        assert not safety_cell_passed(report)


def test_safety_driver_retains_timeout_and_requires_heartbeat_preflight() -> None:
    source = (
        REPO_ROOT / "tools/benchmark/tool_call_latency_benchmark.py"
    ).read_text()
    assert 'except TimeoutError:' in source
    assert '"type": "benchmark.cell_exception"' in source
    assert "cell_dir.mkdir(parents=True, exist_ok=True)" in source
    assert "runtime_provenance = await require_safety_diagnostics(args.url)" in source
    assert 'diagnostics.get("fc_async_heartbeat") is not True' in source
    safety_driver = source[
        source.index("async def run_safety_cell") : source.index("async def run(")
    ]
    assert 'fc.get("active") is True' not in safety_driver
    assert '"type": "benchmark.receive_error"' in safety_driver
    assert '"runner_sha256": sha256_file(Path(__file__))' in source
    assert '"safety_predicate": "safety_cell_passed:v1"' in source
    assert 'parser.add_argument("--prior-safety-report"' in source


def test_reminder_mode_waits_for_preflight_session_cleanup() -> None:
    source = (
        REPO_ROOT / "tools/benchmark/tool_call_latency_benchmark.py"
    ).read_text()
    block = source[
        source.index('if args.mode == "reminder":') : source.index(
            'if args.mode == "same-session":'
        )
    ]
    preflight = block.index("runtime_provenance = await require_safety_diagnostics(args.url)")
    idle = block.index("await wait_for_server_idle(args.url)")
    measured = block.index("cell = await run_call(")
    assert preflight < idle < measured
