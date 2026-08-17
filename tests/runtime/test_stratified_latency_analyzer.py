from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def load_module() -> ModuleType:
    path = Path("tools/qualification/stratified_latency_analyzer.py")
    spec = importlib.util.spec_from_file_location("stratified_latency_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric(
    frame: int,
    *,
    response_id: str | None,
    delivered: bool,
    server_ms: float = 80.0,
    inference_ms: float = 79.0,
    timings: dict | None = None,
    function: dict | None = None,
) -> dict:
    turn_state: dict = {"model_stage_timings_ms": timings}
    if function is not None:
        turn_state["function_calling"] = function
    return {
        "type": "voicechat.metrics",
        "frame": frame,
        "response_id": response_id,
        "audio_delivered": delivered,
        "server_step_ms": server_ms,
        "inference_ms": inference_ms,
        "client_received_monotonic_s": 10.0 + frame / 100.0,
        "output_audio": {"samples": 1_764 if delivered else 0},
        "turn_state": turn_state,
    }


def observation(at: float, samples: int = 1_764) -> dict:
    return {
        "timestamp_s": at,
        "sample_count": samples,
        "sample_rate_hz": 22_050,
    }


def wire_delta(response_id: str, at: float, samples: int = 1_764, **updates: object) -> dict:
    event = {
        "type": "response.output_audio.delta",
        "response_id": response_id,
        "client_received_monotonic_s": at,
        "encoding": "pcm16",
        "sample_rate": 22_050,
        "channels": 1,
        "delta": base64.b64encode(b"\x00\x00" * samples).decode("ascii"),
    }
    event.update(updates)
    return event


def checked_playout_trace() -> list[dict]:
    delta = wire_delta("response-1", 1.1, samples=1_764)
    delta.update(
        {
            "trace_schema": "nemotron_voicechat.playout.v1",
            "ordinal": 1,
            "sample_count": 1_764,
            "sample_rate_hz": 22_050,
        }
    )
    schema = {"trace_schema": "nemotron_voicechat.playout.v1"}
    return [
        {
            **schema,
            "type": "voicechat.playout.config",
            "configured_prebuffer_ms": 160,
            "sample_rate_hz": 22_050,
            "channels": 1,
            "client_configured_monotonic_s": 0.9,
        },
        {
            **schema,
            "type": "response.created",
            "response_id": "response-1",
            "turn_id": "turn-1",
            "client_received_monotonic_s": 1.0,
        },
        delta,
        {
            **schema,
            "type": "response.done",
            "response_id": "response-1",
            "status": "completed",
            "client_received_monotonic_s": 1.2,
        },
        {
            **schema,
            "type": "voicechat.playout.release",
            "response_id": "response-1",
            "reason": "done",
            "ordinals": [1],
            "frames_released": 1,
            "frames_cleared": 0,
            "sample_count": 1_764,
            "client_release_monotonic_s": 1.21,
        },
        {
            **schema,
            "type": "voicechat.playout.downstream_push",
            "response_id": "response-1",
            "ordinal": 1,
            "sample_count": 1_764,
            "sample_rate_hz": 22_050,
            "channels": 1,
            "client_downstream_push_monotonic_s": 1.22,
        },
        {
            **schema,
            "type": "voicechat.playout.trace_status",
            "valid": True,
            "invalid_reason": None,
            "error": None,
            "dropped_records": 0,
            "trace_closed_monotonic_s": 1.3,
        },
    ]


def test_checked_playout_consumer_reconciles_actual_release_and_push_chain() -> None:
    module = load_module()

    validation = module.validate_observed_playout_trace(checked_playout_trace())

    assert validation["observed"] is True
    assert validation["valid"] is True
    assert validation["response_count"] == 1
    assert validation["release_count"] == 1
    assert validation["downstream_push_count"] == 1


def test_checked_playout_consumer_accepts_and_mutation_checks_runtime_provenance() -> None:
    module = load_module()
    trace = checked_playout_trace()
    provenance = {
        "trace_schema": "nemotron_voicechat.playout.v1",
        "type": "session.created.provenance",
        "client_received_monotonic_s": 0.95,
        "session_id": "session-1",
        "session_created_sha256": "a" * 64,
        "runtime_provenance": {
            "runtime_image": "voicechat:hotfix-v1",
            "runtime_image_id": "sha256:" + "b" * 64,
            "runtime_contract": "production-hotfix-notext-watchdog-v1",
            "semantic_environment": {"VOICECHAT_WEB_AGENT_NO_TEXT_FRAMES": "30"},
            "source_sha256": {"server.py": "c" * 64},
        },
    }
    trace.insert(1, provenance)
    assert module.validate_observed_playout_trace(trace)["valid"] is True

    provenance["runtime_provenance"]["runtime_image_id"] = "mutable"
    with pytest.raises(ValueError, match="provenance"):
        module.validate_observed_playout_trace(trace)


@pytest.mark.parametrize("probe", ["review", "duplicate", "orphan", "contradiction"])
def test_checked_playout_consumer_rejects_impossible_records(probe: str) -> None:
    module = load_module()
    trace = checked_playout_trace()
    schema = {"trace_schema": "nemotron_voicechat.playout.v1"}
    if probe == "review":
        trace[-1:-1] = [
            {
                **schema,
                "type": "voicechat.playout.release",
                "response_id": "response-1",
                "reason": "threshold",
                "ordinals": list(range(999)),
                "frames_released": 999,
                "frames_cleared": 999,
                "sample_count": 999,
                "client_release_monotonic_s": 999.0,
            },
            {
                **schema,
                "type": "voicechat.playout.downstream_push",
                "response_id": "wrong-response",
                "ordinal": 999,
                "sample_count": 999,
                "sample_rate_hz": -1,
                "channels": 0,
                "client_downstream_push_monotonic_s": -999.0,
            },
        ]
    elif probe == "duplicate":
        trace.insert(-1, dict(trace[-2]))
    elif probe == "orphan":
        trace[-2]["response_id"] = "wrong-response"
    else:
        trace[-3]["frames_released"] = 2

    with pytest.raises(ValueError):
        module.analyze_events(trace, reserves_ms=(160.0,))


@pytest.mark.parametrize(
    "probe",
    [
        "clockless-release-push",
        "masked-push-regression",
        "unknown-schema-record",
        "done-between-threshold-release-and-push",
        "interruption-after-done-before-release",
    ],
)
def test_checked_playout_consumer_rejects_round2_accepted_contradictions(
    probe: str,
) -> None:
    module = load_module()
    trace = checked_playout_trace()
    schema = {"trace_schema": "nemotron_voicechat.playout.v1"}
    if probe == "clockless-release-push":
        trace[-3].pop("client_release_monotonic_s")
        trace[-2].pop("client_downstream_push_monotonic_s")
    elif probe == "masked-push-regression":
        trace[-2]["client_downstream_push_monotonic_s"] = 0.1
        trace[-2]["client_received_monotonic_s"] = 1.22
    elif probe == "unknown-schema-record":
        trace.insert(
            -1,
            {
                **schema,
                "type": "voicechat.playout.teleport",
                "client_received_monotonic_s": 1.25,
            },
        )
    elif probe == "done-between-threshold-release-and-push":
        trace[0]["configured_prebuffer_ms"] = 80
        done = trace.pop(-4)
        release = trace[-3]
        release["reason"] = "threshold"
        release["client_release_monotonic_s"] = 1.2
        done["client_received_monotonic_s"] = 1.21
        trace.insert(-2, done)
    else:
        trace.insert(
            -3,
            {
                **schema,
                "type": "voicechat.playout.interruption",
                "response_id": "response-1",
                "client_interruption_monotonic_s": 1.205,
            },
        )
        trace[-3].update(
            reason="interruption-clear",
            frames_released=0,
            frames_cleared=1,
            sample_count=0,
        )

    with pytest.raises(ValueError):
        module.analyze_events(trace, reserves_ms=(160.0,))


def test_threshold_release_is_bound_to_traced_prebuffer_identity() -> None:
    module = load_module()
    trace = checked_playout_trace()
    trace[0]["configured_prebuffer_ms"] = 160
    done = trace.pop(3)
    trace[3]["reason"] = "threshold"
    done["client_received_monotonic_s"] = 1.23
    trace.insert(-1, done)

    with pytest.raises(ValueError, match="configured prebuffer"):
        module.validate_observed_playout_trace(trace)


@pytest.mark.parametrize(
    "probe",
    [
        "audio-after-interruption-clear",
        "non-typed-speech-start",
        "zero-sample-chain",
        "self-contradictory-terminal",
    ],
)
def test_checked_playout_consumer_rejects_round3_producer_impossible_traces(
    probe: str,
) -> None:
    module = load_module()
    trace = checked_playout_trace()
    schema = {"trace_schema": "nemotron_voicechat.playout.v1"}
    if probe == "audio-after-interruption-clear":
        done, release, push = trace[3:6]
        del trace[3:6]
        trace[3:3] = [
            {
                **schema,
                "type": "input_audio_buffer.speech_started",
                "source": "typed",
                "response_id": "response-1",
                "client_received_monotonic_s": 1.15,
            },
            {
                **schema,
                "type": "voicechat.playout.release",
                "response_id": "response-1",
                "reason": "interruption-clear",
                "ordinals": [1],
                "frames_released": 0,
                "frames_cleared": 1,
                "sample_count": 0,
                "client_release_monotonic_s": 1.16,
            },
        ]
        second_delta = dict(trace[2])
        second_delta.update(
            ordinal=2,
            client_received_monotonic_s=1.17,
        )
        done["client_received_monotonic_s"] = 1.2
        release.update(
            ordinals=[2],
            sample_count=1_764,
            client_release_monotonic_s=1.21,
        )
        push.update(ordinal=2, client_downstream_push_monotonic_s=1.22)
        trace[5:5] = [second_delta, done, release, push]
    elif probe == "non-typed-speech-start":
        trace.insert(
            3,
            {
                **schema,
                "type": "input_audio_buffer.speech_started",
                "source": "microphone",
                "response_id": "response-1",
                "client_received_monotonic_s": 1.15,
            },
        )
    elif probe == "zero-sample-chain":
        trace[2].update(delta="", sample_count=0)
        trace[4]["sample_count"] = 0
        trace[5]["sample_count"] = 0
    else:
        trace[-1].update(invalid_reason="writer-error", error="OSError('disk')")

    with pytest.raises(ValueError):
        module.validate_observed_playout_trace(trace)


@pytest.mark.parametrize("probe", ["extraneous-field", "bool-ordinal"])
def test_checked_playout_consumer_enforces_exact_fields_and_json_scalar_types(
    probe: str,
) -> None:
    module = load_module()
    trace = checked_playout_trace()
    if probe == "extraneous-field":
        trace[1]["unexpected"] = "bypass"
    else:
        trace[2]["ordinal"] = True

    with pytest.raises(ValueError, match="schema"):
        module.validate_observed_playout_trace(trace)


def test_population_predicates_are_disjoint_and_high_modes_are_delivered_nonbos() -> None:
    module = load_module()
    bos_timings = {
        "nano_interface_ms": 50.0,
        "eartts_total_ms": 80.0,
        "wrapper_residual_ms": 12.0,
        "eartts_reset": {"reset_total_ms": 70.0},
    }
    steady_timings = {
        "nano_interface_ms": 52.0,
        "eartts_total_ms": 18.0,
        "wrapper_residual_ms": 9.0,
        "eartts_reset": None,
    }
    events = [
        metric(1, response_id="response-1", delivered=True, timings=bos_timings),
        metric(2, response_id="response-1", delivered=True, timings=steady_timings),
        metric(3, response_id=None, delivered=False, timings=steady_timings),
        metric(
            4,
            response_id=None,
            delivered=False,
            timings=None,
            function={"active": True, "background_active": True},
        ),
    ]

    populations = module.classify_populations(module._normalize_model_rows(events))

    assert {name: len(rows) for name, rows in populations.items()} == {
        "delivered": 2,
        "delivered-nonBOS": 1,
        "BOS-transition": 1,
        "abort-prefill-reset": 1,
        "response-null": 2,
        "idle": 1,
        "function-cycle": 1,
        "EarTTS-high": 1,
        "residual-high": 0,
    }
    delivered = module._population_report(populations["delivered"])
    assert delivered["clocks"]["server_step_ms"]["mean"] == 80.0
    assert delivered["stage_timings"]["nano_interface"] == {"n": 2, "mean_ms": 51.0}
    assert delivered["membership"]["n"] == 2
    assert len(delivered["membership"]["sha256"]) == 64


def test_prepared_reuse_is_bos_transition_but_not_abort_prefill_reset() -> None:
    module = load_module()
    event = metric(
        1,
        response_id="response-1",
        delivered=True,
        timings={"eartts_total_ms": 12.0, "eartts_reset": None},
    )
    event["turn_state"]["eartts_bos_epoch_step"] = {
        "transition_count_before": 1,
        "transition_count_after": 2,
        "reset_count_before": 1,
        "reset_count_after": 1,
        "reuse_count_before": 0,
        "reuse_count_after": 1,
    }

    populations = module.classify_populations(module._normalize_model_rows([event]))

    assert len(populations["BOS-transition"]) == 1
    assert populations["abort-prefill-reset"] == []
    assert populations["delivered-nonBOS"] == []


def test_response_null_is_literal_and_idle_excludes_delivered_rows() -> None:
    module = load_module()
    delivered_without_field = {
        "event": "model_step",
        "audio_delivered": True,
        "monotonic_s": 1.0,
        "output_audio": {"bytes": 3_528, "samples": 1_764},
        "turn_state": {"model_stage_timings_ms": {}},
    }
    idle = {
        "event": "model_step",
        "audio_delivered": False,
        "monotonic_s": 1.1,
        "output_audio": {"bytes": 0, "samples": 0},
        "turn_state": {"model_stage_timings_ms": {}},
    }
    function_cycle = {
        "event": "model_step",
        "audio_delivered": False,
        "monotonic_s": 1.2,
        "output_audio": {"bytes": 0, "samples": 0},
        "turn_state": {"function_calling": {"awaiting_response": True}},
    }

    rows = module._normalize_model_rows([delivered_without_field, idle, function_cycle])
    populations = module.classify_populations(rows)

    assert rows[0]["response_group"] == "inferred-response-0001"
    assert len(populations["response-null"]) == 3
    assert populations["idle"] == [rows[1]]
    assert populations["function-cycle"] == [rows[2]]


def test_audio_delivered_is_canonical_and_bytes_are_only_a_legacy_fallback() -> None:
    module = load_module()
    retained_but_not_delivered = {
        "event": "model_step",
        "audio_delivered": False,
        "monotonic_s": 1.0,
        "output_audio": {"bytes": 3_528, "samples": 1_764},
        "turn_state": {"model_stage_timings_ms": {}},
    }
    legacy = {
        "event": "model_step",
        "monotonic_s": 1.1,
        "output_audio": {"bytes": 3_528, "samples": 1_764},
        "turn_state": {"model_stage_timings_ms": {}},
    }

    rows = module._normalize_model_rows([retained_but_not_delivered, legacy])

    assert rows[0]["delivered"] is False
    assert rows[0]["delivery_mode"] == "audio_delivered"
    assert rows[1]["delivered"] is True
    assert rows[1]["delivery_mode"] == "legacy_output_audio_bytes"


def test_nearest_rank_boundaries_do_not_interpolate() -> None:
    module = load_module()

    twenty = [float(value) for value in range(1, 21)]
    twenty_one = [float(value) for value in range(1, 22)]
    assert module.nearest_rank(twenty, 95) == 19.0
    assert module.nearest_rank(twenty_one, 95) == 20.0
    assert module.nearest_rank(twenty, 99) == 20.0
    assert module.nearest_rank(twenty_one, 99) == 21.0
    assert module.nearest_rank([], 50) is None
    with pytest.raises(ValueError, match="percentile"):
        module.nearest_rank([1.0], 0)


def test_release_threshold_recovers_queue_and_labels_starvation_as_simulated() -> None:
    module = load_module()
    deltas = [observation(0.0), observation(0.100), observation(0.400), observation(0.470)]

    replay = module.replay_response(deltas, 160.0, done_time_s=0.471)

    assert replay["release_reason"] == "threshold"
    assert replay["release_time_s"] == pytest.approx(0.100)
    assert replay["added_release_delay_ms"] == pytest.approx(100.0)
    assert replay["simulated_starvation_events"] == 1
    assert replay["starvation_ms"] == pytest.approx(140.0)
    assert replay["downstream_queue_ms_at_stop"] == pytest.approx(90.0)
    assert "not browser underrun" in replay["starvation_evidence"]


def test_done_flush_releases_only_when_response_done_was_observed() -> None:
    module = load_module()
    deltas = [observation(2.0, samples=882)]

    observed = module.replay_response(deltas, 160.0, done_time_s=2.125, done_observed=True)
    unknown = module.replay_response(deltas, 160.0, done_observed=False)

    assert observed["released"] is True
    assert observed["release_reason"] == "response.done"
    assert observed["added_release_delay_ms"] == pytest.approx(125.0)
    assert unknown["released"] is False
    assert unknown["release_reason"] == "done_flush_unknown"
    assert unknown["added_release_delay_ms"] is None


def test_interruption_ordering_models_clear_and_detached_release_burst() -> None:
    module = load_module()
    deltas = [observation(0.0), observation(0.080), observation(1.000)]

    pre_arrival = module.replay_response(
        deltas,
        160.0,
        interruption_point="pre-arrival",
        interruption_ordinal=1,
    )
    pre_release = module.replay_response(
        deltas,
        160.0,
        interruption_point="post-buffer/pre-release",
        interruption_ordinal=1,
    )
    in_burst = module.replay_response(
        deltas,
        160.0,
        interruption_point="post-release-frame",
        interruption_ordinal=0,
    )
    uninterrupted = module.replay_response(deltas, 160.0, done_observed=False)

    assert pre_arrival["processed_delta_count"] == 1
    assert pre_arrival["service_held_media_ms_at_stop"] == 0.0
    assert pre_arrival["released_downstream_ms"] == 0.0
    assert pre_release["processed_delta_count"] == 2
    assert pre_release["released"] is False
    assert pre_release["service_held_media_ms_at_stop"] == 0.0
    assert in_burst["released_downstream_ms_at_interrupt"] == pytest.approx(80.0)
    assert in_burst["release_burst_after_interrupt_ms"] == pytest.approx(80.0)
    assert in_burst["downstream_queue_ms_at_stop"] == pytest.approx(160.0)
    assert uninterrupted["starvation_ms"] == pytest.approx(760.0)


def test_at_release_interruption_exposes_the_whole_detached_burst() -> None:
    module = load_module()
    deltas = [observation(0.0), observation(0.080)]

    replay = module.replay_response(
        deltas,
        160.0,
        interruption_point="at-release/post-detach/pre-first-push",
        interruption_ordinal=1,
    )

    assert replay["released"] is True
    assert replay["released_downstream_ms_at_interrupt"] == 0.0
    assert replay["release_burst_after_interrupt_ms"] == pytest.approx(160.0)
    assert replay["downstream_queue_ms_at_stop"] == pytest.approx(160.0)


def test_pre_arrival_interruption_drains_elapsed_downstream_playout() -> None:
    module = load_module()
    deltas = [observation(0.0), observation(0.080), observation(0.180)]

    replay = module.replay_response(
        deltas,
        160.0,
        interruption_point="pre-arrival",
        interruption_ordinal=2,
    )

    assert replay["released_downstream_ms"] == pytest.approx(160.0)
    assert replay["downstream_queue_ms_at_interrupt"] == pytest.approx(60.0)
    assert replay["downstream_queue_ms_at_stop"] == pytest.approx(60.0)


def test_interruption_sweep_contains_all_required_ordering_points() -> None:
    module = load_module()
    deltas = [observation(0.0), observation(0.080), observation(1.000)]
    diagnostics = module.queue_diagnostics(
        {
            "response-1": {
                "response_id": "response-1",
                "deltas": deltas,
                "done_time_s": 1.001,
                "done_observed": True,
                "clock_normative": True,
            }
        },
        (160.0,),
    )

    sweep = diagnostics["responses"][0]["reserve_replays"][0]["interruption_sweep"]
    points = {(item["interruption_point"], item["interruption_ordinal"]) for item in sweep}
    assert ("pre-arrival", 0) in points
    assert ("post-buffer/pre-release", 1) in points
    assert ("at-release/post-detach/pre-first-push", 1) in points
    assert ("post-release-frame", 0) in points
    assert ("post-release-frame", 2) in points


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"encoding": "opus"}, "encoding"),
        ({"sample_rate": 24_000}, "mono PCM16"),
        ({"channels": 2}, "mono PCM16"),
        ({"sample_rate": None}, "mono PCM16"),
    ],
)
def test_audio_delta_wire_contract_matches_runtime_decoder(updates: dict, message: str) -> None:
    module = load_module()
    events = [
        {"type": "response.created", "response_id": "response-1"},
        wire_delta("response-1", 1.0, **updates),
    ]

    with pytest.raises(ValueError, match=message):
        module.extract_queue_responses(events, rows=[])


def test_stale_and_cancelling_deltas_are_ignored_before_wire_decode() -> None:
    module = load_module()
    malformed_stale = wire_delta("stale", 0.9, encoding="opus")
    malformed_cancelling = wire_delta("response-1", 1.2, encoding="opus")
    events = [
        malformed_stale,
        {"type": "response.created", "response_id": "response-1"},
        wire_delta("response-1", 1.0, samples=3),
        {"type": "input_audio_buffer.speech_started", "source": "typed"},
        malformed_cancelling,
        {
            "type": "response.done",
            "response_id": "response-1",
            "client_received_monotonic_s": 1.3,
        },
    ]

    _, responses, audit = module.extract_queue_responses(events, rows=[])

    assert responses["response-1"]["deltas"][0]["sample_count"] == 3
    assert len(responses["response-1"]["deltas"]) == 1
    assert audit["ignored_stale_delta_count"] == 1
    assert audit["ignored_cancelling_delta_count"] == 1


def test_non_typed_speech_started_does_not_cancel_the_response() -> None:
    module = load_module()
    events = [
        {"type": "response.created", "response_id": "response-1"},
        {"type": "input_audio_buffer.speech_started", "source": "microphone"},
        wire_delta("response-1", 1.0, samples=3),
    ]

    _, responses, audit = module.extract_queue_responses(events, rows=[])

    assert responses["response-1"]["deltas"][0]["sample_count"] == 3
    assert responses["response-1"]["clear_events"] == []
    assert audit["ignored_cancelling_delta_count"] == 0
    assert audit["observed_interruption_clear_count"] == 0


def test_typed_clear_then_done_does_not_resurrect_service_held_media() -> None:
    module = load_module()
    events = [
        {"type": "response.created", "response_id": "response-1"},
        wire_delta("response-1", 1.0, samples=882),
        {
            "type": "input_audio_buffer.speech_started",
            "source": "typed",
            "client_received_monotonic_s": 1.1,
        },
        {
            "type": "response.done",
            "response_id": "response-1",
            "client_received_monotonic_s": 1.2,
        },
    ]

    _, responses, audit = module.extract_queue_responses(events, rows=[])
    diagnostics = module.queue_diagnostics(responses, (160.0,))
    response = diagnostics["responses"][0]
    replay = response["reserve_replays"][0]

    assert response["done_observed"] is True
    assert audit["observed_interruption_clear_count"] == 1
    assert replay["observed_clear_count"] == 1
    assert replay["cleared_service_held_media_ms"] == pytest.approx(40.0)
    assert replay["released"] is False
    assert replay["release_reason"] == "interruption-clear"
    assert replay["released_downstream_ms"] == 0.0
    assert replay["service_held_media_ms_at_stop"] == 0.0


def test_observed_clear_preserves_only_elapsed_downstream_exposure() -> None:
    module = load_module()
    events = [
        {"type": "response.created", "response_id": "response-1"},
        wire_delta("response-1", 1.0),
        wire_delta("response-1", 1.080),
        {
            "type": "input_audio_buffer.speech_started",
            "source": "typed",
            "client_received_monotonic_s": 1.180,
        },
        {
            "type": "response.done",
            "response_id": "response-1",
            "client_received_monotonic_s": 1.2,
        },
    ]

    _, responses, _ = module.extract_queue_responses(events, rows=[])
    replay = module.queue_diagnostics(responses, (160.0,))["responses"][0]["reserve_replays"][0]

    assert replay["released"] is True
    assert replay["release_reason"] == "threshold"
    assert replay["cleared_service_held_media_ms"] == 0.0
    assert replay["released_downstream_ms"] == pytest.approx(160.0)
    assert replay["downstream_queue_ms_at_interrupt"] == pytest.approx(60.0)
    assert replay["release_burst_after_interrupt_ms"] == 0.0


def test_server_only_short_response_never_fabricates_done_flush() -> None:
    module = load_module()
    events = [
        {
            "event": "model_step",
            "audio_delivered": True,
            "monotonic_s": 1.0,
            "output_audio": {"bytes": 1_764, "samples": 882},
            "turn_state": {"model_stage_timings_ms": {}},
        }
    ]
    rows = module._normalize_model_rows(events)

    _, responses, _ = module.extract_queue_responses(events, rows)
    diagnostics = module.queue_diagnostics(responses, (160.0,))
    replay = diagnostics["responses"][0]["reserve_replays"][0]

    assert diagnostics["normative"] is False
    assert diagnostics["responses"][0]["done_observed"] is False
    assert replay["released"] is False
    assert replay["release_reason"] == "done_flush_unknown"


def test_literal_k12_boundary_gap_bound_and_insufficient_coverage() -> None:
    module = load_module()
    exactly_960 = [index * 0.960 / 11 for index in range(12)]

    accepted = module.structural_gate({"response-1": exactly_960}, max_gap_ms=160.0)
    failed_window = module.structural_gate(
        {"response-1": exactly_960[:-1] + [0.960002]}, max_gap_ms=160.0
    )
    failed_gap = module.structural_gate(
        {"response-1": [index * 0.08 for index in range(11)] + [0.961]},
        max_gap_ms=160.0,
    )
    insufficient = module.structural_gate({"response-1": [index * 0.08 for index in range(11)]})

    assert accepted["window_count"] == 1
    assert accepted["verdict"] == "ACCEPT"
    assert failed_window["verdict"] == "REJECT"
    assert failed_gap["max_gap_ms"] == pytest.approx(161.0)
    assert failed_gap["verdict"] == "REJECT"
    assert insufficient["window_count"] == 0
    assert insufficient["verdict"] == "INSUFFICIENT"
    assert insufficient["qualification_eligible"] is False


def test_structural_gate_uses_reconciled_delta_clock_and_metrics_is_diagnostic() -> None:
    module = load_module()
    events: list[dict] = [{"type": "response.created", "response_id": "response-1"}]
    for index in range(12):
        events.extend(
            [
                wire_delta("response-1", index * 0.100),
                metric(index, response_id="response-1", delivered=True, timings={}),
            ]
        )
    events.append(
        {
            "type": "response.done",
            "response_id": "response-1",
            "client_received_monotonic_s": 1.2,
        }
    )

    report = module.analyze_events(events, reserves_ms=(160.0,))
    gate = report["structural_gate"]
    diagnostic = gate["diagnostics"]["metrics_receipt"]

    assert gate["clock"].endswith("(response.output_audio.delta)")
    assert gate["reconciliation"]["provable"] is True
    assert gate["worst_window"]["duration_ms"] == pytest.approx(1_100.0)
    assert gate["verdict"] == "REJECT"
    assert diagnostic["normative"] is False
    assert diagnostic["diagnostic_outcome"] == "ACCEPT"
    assert diagnostic["verdict"] is None


def test_unreconciled_delta_count_makes_normative_structural_gate_unavailable() -> None:
    module = load_module()
    events = [
        {"type": "response.created", "response_id": "response-1"},
        wire_delta("response-1", 1.0),
        metric(1, response_id="response-1", delivered=True, timings={}),
        metric(2, response_id="response-1", delivered=True, timings={}),
    ]

    gate = module.analyze_events(events, reserves_ms=(160.0,))["structural_gate"]

    assert gate["reconciliation"]["provable"] is False
    assert gate["qualification_eligible"] is False
    assert gate["verdict"] == "UNAVAILABLE"


def test_zero_reserve_prefix_debt_is_cumulative_not_nonrecovering_sum() -> None:
    module = load_module()
    deltas = [observation(0.0), observation(0.100), observation(0.170), observation(0.280)]

    debt = module.max_prefix_debt(deltas)

    assert debt["max_prefix_debt_ms"] == pytest.approx(40.0)
    assert debt["maximum_after_delta_ordinal"] == 3


def test_synthetic_analyze_cli_writes_identity_and_normative_status(tmp_path: Path) -> None:
    module = load_module()
    events = [
        {"type": "response.created", "response_id": "response-1"},
        wire_delta("response-1", 10.01),
        metric(
            1,
            response_id="response-1",
            delivered=True,
            server_ms=81.0,
            inference_ms=79.0,
            timings={"eartts_total_ms": 12.0, "wrapper_residual_ms": 4.0},
        ),
        {
            "type": "response.done",
            "response_id": "response-1",
            "client_received_monotonic_s": 10.02,
        },
    ]
    source = tmp_path / "events.jsonl"
    source.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    output = tmp_path / "report.json"

    result = module.main(
        ["analyze", str(source), "--output", str(output), "--reserve-sweep", "0,160"]
    )
    analyzed = json.loads(output.read_text(encoding="utf-8"))["sources"][0]

    assert result == 0
    assert analyzed["source"]["path"] == str(source.resolve())
    assert analyzed["source"]["bytes"] == source.stat().st_size
    assert len(analyzed["source"]["sha256"]) == 64
    assert analyzed["populations"]["delivered"]["clocks"]["server_step_ms"]["mean"] == 81.0
    assert analyzed["queue_diagnostics"]["normative"] is True
    assert analyzed["structural_gate"]["verdict"] == "INSUFFICIENT"


def test_integrity_checks_fail_on_wrong_source_or_membership_hash() -> None:
    module = load_module()
    reports = {
        "fixture": {
            "source": {"sha256": "source-hash"},
            "populations": {"delivered": {"membership": {"sha256": "membership-hash"}}},
        }
    }

    valid = module._integrity_checks(
        reports,
        {"fixture": "source-hash"},
        {("fixture", "delivered"): "membership-hash"},
    )
    wrong_source = module._integrity_checks(
        reports,
        {"fixture": "wrong-source-hash"},
        {("fixture", "delivered"): "membership-hash"},
    )
    wrong_membership = module._integrity_checks(
        reports,
        {"fixture": "source-hash"},
        {("fixture", "delivered"): "wrong-membership-hash"},
    )

    assert all(check["passed"] for check in valid)
    assert not all(check["passed"] for check in wrong_source)
    assert not all(check["passed"] for check in wrong_membership)


def test_default_aged_anchor_path_is_durable() -> None:
    module = load_module()

    path = module._canonical_anchor_paths(None)["aged"]

    assert "/.local/state/nemotron-voicechat/qualification/" in str(path)
    assert "sustained-aged-26h-20260812" in str(path)
    assert "/tmp/" not in str(path)


def test_step1_details_use_token_and_request_position_evidence(tmp_path: Path) -> None:
    module = load_module()

    def server_row(
        frame: int,
        *,
        token: int | None,
        nano_position: int,
        nano_ms: float,
        eartts_ms: float = 12.0,
        residual_ms: float = 4.0,
        bos: bool = False,
        delivered: bool = True,
    ) -> dict:
        timings = {
            "perception_ms": 12.0,
            "nano_interface_ms": nano_ms,
            "eartts_total_ms": eartts_ms,
            "codec_call_ms": 0.7,
            "wrapper_residual_ms": residual_ms,
            "wrapper_total_ms": 80.0,
            "eartts_reset": {"reset_total_ms": 60.0} if bos else None,
        }
        return {
            "event": "model_step",
            "frame": frame,
            "audio_delivered": delivered,
            "monotonic_s": 1.0 + frame * 0.08,
            "server_step_ms": 81.0 + frame,
            "inference_ms": 80.0 + frame,
            "output_audio": {
                "bytes": 3_528 if delivered else 0,
                "samples": 1_764 if delivered else 0,
            },
            "turn_state": {
                "agent_token_id": token,
                "agent_control": "agent_bos" if bos else ("pad" if token == 12 else None),
                "control_ids": {"agent_bos": 1, "agent_eos": 2, "pad": 12},
                "model_stage_timings_ms": timings,
                "vllm_request_positions": {
                    "request_id": "request-1",
                    "nano": {"session_positions": nano_position},
                },
            },
        }

    pair_events = [
        server_row(1, token=1, nano_position=1, nano_ms=52.0, bos=True),
        server_row(2, token=12, nano_position=1, nano_ms=0.2),
        server_row(3, token=12, nano_position=3, nano_ms=64.0, eartts_ms=19.0),
        server_row(4, token=2, nano_position=4, nano_ms=52.0),
    ]
    pair_events[-1]["turn_state"]["agent_control"] = "agent_eos"
    sequential_events = [
        server_row(1, token=1, nano_position=1, nano_ms=52.0, bos=True),
        server_row(2, token=900, nano_position=2, nano_ms=52.0, residual_ms=11.0),
        server_row(3, token=12, nano_position=3, nano_ms=52.0),
        server_row(4, token=2, nano_position=4, nano_ms=52.0),
        server_row(5, token=12, nano_position=5, nano_ms=52.0, delivered=False),
        server_row(6, token=None, nano_position=6, nano_ms=52.0),
    ]
    sequential_events[-2]["turn_state"]["agent_control"] = None
    sequential_events[-2]["turn_state"]["response_boundary"] = {
        "phase": "internal_drain"
    }
    pair_path = tmp_path / "pair" / "events.jsonl"
    sequential_path = tmp_path / "sequential" / "events.jsonl"
    pair_path.parent.mkdir()
    sequential_path.parent.mkdir()
    pair_path.write_text(
        "".join(json.dumps(event) + "\n" for event in pair_events), encoding="utf-8"
    )
    sequential_path.write_text(
        "".join(json.dumps(event) + "\n" for event in sequential_events), encoding="utf-8"
    )

    report = module.analyze_paths([pair_path, sequential_path], step1_details=True)
    details = report["step1_details"]

    assert details["artifact_count"] == 2
    phase_strata = details["stage_timing_strata"]
    assert phase_strata["delivered-nonBOS-stage-timed"]["membership"]["n"] == 7
    assert details["stage_timing_strata"]["text-emission"]["membership"]["n"] == 1
    assert details["stage_timing_strata"]["PAD-tail"]["membership"]["n"] == 3
    assert details["stage_timing_strata"]["agent-control"]["membership"]["n"] == 2
    assert details["stage_timing_strata"]["token-unavailable"]["membership"]["n"] == 1
    assert sum(
        phase_strata[name]["membership"]["n"]
        for name in ("text-emission", "PAD-tail", "agent-control", "token-unavailable")
    ) == phase_strata["delivered-nonBOS-stage-timed"]["membership"]["n"]
    phase_members = {
        (member["artifact"], member["frame"])
        for name in ("text-emission", "PAD-tail", "agent-control", "token-unavailable")
        for member in phase_strata[name]["membership"]["members"]
    }
    assert ("sequential", 5) not in phase_members
    assert ("pair", 1) not in phase_members
    assert ("sequential", 1) not in phase_members
    assert details["stage_timing_strata"]["abort-prefill-reset"]["membership"]["n"] == 2
    assert details["stage_timing_strata"]["BOS-transition"]["membership"]["n"] == 2
    comparison = details["pair_vs_sequential"]
    assert comparison["pair_artifact"] == "pair"
    assert comparison["pair_schedule"]["classes"]["buffered"]["membership"]["n"] == 1
    assert comparison["pair_schedule"]["classes"]["packed-pair"]["membership"]["n"] == 1
    assert comparison["pair_schedule"]["pair_class_membership"]["n"] == 2
    assert comparison["pair_schedule"]["transitions"] == {"buffered->packed-pair": 1}
    high = details["high_mode_inventories"]["EarTTS-high"]
    assert high["membership"]["n"] == 1
    assert high["rows"][0]["phase"] == "PAD-tail"
    assert high["rows"][0]["previous_model_row"]["frame"] == 2
    assert high["rows"][0]["next_model_row"]["phase"] == "agent-control"
    assert high["rows"][0]["stage_timings_ms"]["eartts_total"] == 19.0
    residual = details["high_mode_inventories"]["residual-high"]
    assert residual["membership"]["n"] == 1
    assert residual["rows"][0]["phase"] == "text-emission"
    markdown = module.render_step1_markdown(report)
    assert "## Corpus inventory" in markdown
    assert "buffered→packed 1" in markdown
    assert pair_events[0]["turn_state"]["model_stage_timings_ms"]["eartts_reset"]
    assert report["sources"][0]["source"]["sha256"] in markdown


def test_step1_cli_flags_write_json_and_markdown(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "events.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    calls: dict = {}

    def analyze_paths(paths: list[Path], **kwargs: object) -> dict:
        calls["paths"] = paths
        calls.update(kwargs)
        return {"sources": [], "step1_details": {}}

    module.analyze_paths = analyze_paths
    module.render_step1_markdown = lambda report: "# rendered\n"

    result = module.main(
        [
            "analyze",
            str(source),
            "--output",
            str(output),
            "--step1-details",
            "--markdown-output",
            str(markdown_output),
        ]
    )

    assert result == 0
    assert calls["paths"] == [source]
    assert calls["step1_details"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["step1_details"] == {}
    assert markdown_output.read_text(encoding="utf-8") == "# rendered\n"


def test_markdown_output_without_step1_details_fails_before_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_module()
    source = tmp_path / "events.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    module.analyze_paths = lambda *args, **kwargs: pytest.fail("analysis must not start")

    with pytest.raises(SystemExit, match="2"):
        module.main(
            [
                "analyze",
                str(source),
                "--output",
                str(output),
                "--markdown-output",
                str(markdown_output),
            ]
        )

    assert "--markdown-output requires --step1-details" in capsys.readouterr().err
    assert not output.exists()
    assert not markdown_output.exists()


def test_active_metrics_frame_zero_session9_artifact_validates() -> None:
    """Real sessions begin at frame 0 (session-9 live artifact regression)."""

    import hashlib
    from pathlib import Path

    artifact = Path(
        "reports/step2-live/session-20260813T082710Z/driver/pipecat-playout-0001.jsonl"
    )
    if not artifact.exists():
        pytest.skip("retained session-9 artifact not present on this machine")
    payload = artifact.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    module = load_module()
    events = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    frames = [
        event.get("frame")
        for event in events
        if event.get("type") == "voicechat.metrics" and "audio_delivered" in event
    ]
    assert 0 in frames, "regression requires a frame-0 active metrics row"
    validation = module.validate_observed_playout_trace(events)
    assert validation["valid"] is True, validation
    assert len(digest) == 64
