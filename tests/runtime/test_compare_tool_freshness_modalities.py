from __future__ import annotations

import copy
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from nemotron_voicechat_runtime.artifacts import sha256_file
from nemotron_voicechat_runtime.tool_freshness_contract import (
    derive_structural_conjuncts,
    evaluate_tool_cycle_cardinality,
    evaluate_tool_response_channels,
)


def load_qualification_module(name: str):
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        path = Path(f"tools/qualification/{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def base_report() -> dict:
    path = Path("tests/runtime/test_transcribe_sustained.py")
    spec = importlib.util.spec_from_file_location("transcribe_test_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.complete_tool_freshness_report()


def write_tone(path: Path, sample_rate: int) -> None:
    samples = (
        np.sin(np.arange(sample_rate // 2) * 2 * np.pi * 220 / sample_rate) * 0.1 * 32767
    ).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def evaluator_provenance(comparator) -> dict:
    return {
        **comparator.expected_asr_contract(),
        **comparator.ASR_PLATFORM_CONTRACT,
        "evaluator_image_id": comparator.PROMOTED_ASR_IMAGE_ID,
    }


def prepare_arm(
    root: Path,
    *,
    modality: str,
    model_semantics: tuple[bool, bool, bool, bool],
    audio_semantics: tuple[bool, bool, bool, bool],
) -> Path:
    comparator = load_qualification_module("compare_tool_freshness_modalities")
    transcriber = load_qualification_module("transcribe_sustained")
    report = base_report()
    report["input_modality"] = modality
    report["model_input_path"] = (
        "server_typed_pocket" if modality == "typed" else "client_pocket_fixture_pcm"
    )
    report["session_id"] = f"session-{modality}"
    report["session_configuration"]["id"] = report["session_id"]
    arm = root / modality
    arm.mkdir(parents=True)
    for ordinal, (input_record, response, model_match, audio_match) in enumerate(
        zip(
            report["inputs"],
            report["responses"],
            model_semantics,
            audio_semantics,
            strict=True,
        ),
        1,
    ):
        code = response["expected_response_any"][0]
        response["text"] = (
            f"{code} fresh word is ready now." if model_match else "The fresh word is ready now."
        )
        source = arm / f"response-{ordinal:03d}.wav"
        canonical = arm / f"response-{ordinal:03d}-16k.wav"
        write_tone(source, 22_050)
        source_values, _rate = transcriber.read_and_resample(source)
        transcriber.write_pcm16(canonical, source_values)
        response["audio_path"] = str(source)
        response["audio_bytes"] = source.stat().st_size - 44
        transcript = (
            f"{code} fresh word is ready now." if audio_match else "The fresh word is ready now."
        )
        evaluation = transcriber.response_asr_evaluation(
            hypothesis=transcript,
            reference=response["text"],
            expected=[code],
            excluded=[],
            max_wer=0.5,
            parity_arm=True,
        )
        values, _rate = transcriber.read_and_resample(source)
        classification, level = transcriber.classify_audio(values)
        assert classification == "speech" and evaluation["audio_integrity_passed"]
        response["external_asr"] = {
            "classification": classification,
            "audio_dbfs": level,
            "wav_16k": str(canonical),
            "source_audio_sha256": sha256_file(source),
            "canonical_wav_sha256": sha256_file(canonical),
            "transcript": transcript,
            "token_ids": [ordinal, ordinal + 10],
            **evaluation,
        }
        input_record["input_modality"] = modality
        if modality == "speech":
            input_record.update(
                {
                    "requested_text": None,
                    "client_turn_id": ordinal,
                    "speech_started": True,
                    "turn_started_ack": True,
                    "committed_ack": True,
                    "turn_started_turn_id": input_record["turn_id"],
                    "committed_turn_id": input_record["turn_id"],
                    "transcription_source": "microphone",
                    "transcription_job_id": None,
                    "transcription_turn_id": input_record["turn_id"],
                    "displayed_input_text": input_record["expected_input_text"],
                    "input_wer": 0.0,
                    "event_counts": {
                        "input_audio_buffer.speech_started": 1,
                        "input_audio_buffer.turn_started": 1,
                        "input_audio_buffer.committed": 1,
                        "conversation.item.input_audio_transcription.completed": 1,
                    },
                    "event_correlations": [True] * 4,
                }
            )
    checks, _semantic_coupled = evaluate_tool_response_channels(
        report["responses"], report["metrics"], report["eotr_observations"]
    )
    cardinality = evaluate_tool_cycle_cardinality(
        requested_tool_prompts=4,
        tool_response_checks=checks,
        metrics=report["metrics"],
    )
    conjuncts = derive_structural_conjuncts(
        modality=modality,
        inputs=report["inputs"],
        responses=report["responses"],
        checks=checks,
        cardinality=cardinality,
        errors=report["errors"],
        session_closed=report["session_closed"],
        scenario=report["scenario"],
        fixtures=report["fixture_manifest"]["fixtures"],
        scenario_identity=True,
        runtime_fixture_identity=True,
    )
    assert all(conjuncts.values())
    report["tool_response_checks"] = checks
    report["tool_cycle_cardinality"] = cardinality
    report["structural_conjuncts"] = conjuncts
    report["structural_passed"] = True
    report["runtime_gate_passed"] = True
    provenance = evaluator_provenance(comparator)
    report["external_asr"] = {
        **provenance,
        "response_count": 4,
        "speech_responses": 4,
        "passing_speech_responses": 4,
        "near_silent_responses": 0,
        "near_silent_rate": 0.0,
        "qualified_fp32_base_rate": 0.0,
        "max_wer": 0.5,
        "passed": True,
        "evaluation_policy": comparator.ASR_POLICY,
    }
    report["passed"] = True
    assert transcriber.tool_freshness_runtime_gate_passed(report)
    raw_report = copy.deepcopy(report)
    raw_report.pop("external_asr")
    raw_report.pop("runtime_gate_passed")
    for response in raw_report["responses"]:
        response.pop("external_asr")
        response.pop("audio_dbfs", None)
    raw_path = root / "runtime" / modality / "report.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["external_asr_source_report"] = {"sha256": sha256_file(raw_path)}
    path = arm / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def prepare_pair(
    tmp_path: Path,
    *,
    typed_model=(False, False, False, False),
    speech_model=(False, False, False, False),
    typed_audio=(False, False, False, False),
    speech_audio=(False, False, False, False),
) -> tuple[object, Path, Path, Path]:
    comparator = load_qualification_module("compare_tool_freshness_modalities")
    typed = prepare_arm(
        tmp_path,
        modality="typed",
        model_semantics=typed_model,
        audio_semantics=typed_audio,
    )
    speech = prepare_arm(
        tmp_path,
        modality="speech",
        model_semantics=speech_model,
        audio_semantics=speech_audio,
    )
    typed_report = json.loads(typed.read_text(encoding="utf-8"))
    speech_report = json.loads(speech.read_text(encoding="utf-8"))
    runtime = {
        "schema": 1,
        "kind": "tool_freshness_parity_runtime",
        "comparison_policy": "parity_margin_v2",
        "execution_order": ["typed", "speech"],
        "scenario": typed_report["scenario"],
        "scenario_sha256": typed_report["scenario_sha256"],
        "fixture_manifest_sha256": typed_report["fixture_manifest"]["sha256"],
        "experiment_sha256": typed_report["experiment_sha256"],
        "session_ids": [typed_report["session_id"], speech_report["session_id"]],
        "runtime_provenance": typed_report["runtime_provenance"],
        "arms": {
            modality: {
                "report": str(tmp_path / f"runtime/{modality}/report.json"),
                "structural_passed": True,
            }
            for modality in ("typed", "speech")
        },
        "passed": True,
    }
    runtime_path = tmp_path / "runtime/report.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparator, typed, speech, runtime_path


def set_first_response_transcript(
    root: Path,
    paths: tuple[Path, Path],
    *,
    text: str,
    transcript: str,
) -> None:
    transcriber = load_qualification_module("transcribe_sustained")
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        response = report["responses"][0]
        response["text"] = text
        evaluation = transcriber.response_asr_evaluation(
            hypothesis=transcript,
            reference=text,
            expected=response["expected_response_any"],
            excluded=[],
            max_wer=0.5,
            parity_arm=True,
        )
        response["external_asr"].update({"transcript": transcript, **evaluation})
        modality = report["input_modality"]
        raw = root / f"runtime/{modality}/report.json"
        raw_report = json.loads(raw.read_text(encoding="utf-8"))
        raw_report["responses"][0]["text"] = text
        raw.write_text(json.dumps(raw_report), encoding="utf-8")
        report["external_asr_source_report"]["sha256"] = sha256_file(raw)
        path.write_text(json.dumps(report), encoding="utf-8")


def test_shared_semantic_failures_are_reported_but_do_not_block(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)

    artifact = comparator.compare(typed, speech, runtime)

    assert artifact["passed"] is True
    assert not artifact["blocking_reasons"]
    assert all(case["model_text_semantic"]["shared_failure"] for case in artifact["comparisons"])
    assert all(
        case["response_audio_semantic"]["shared_failure"] for case in artifact["comparisons"]
    )
    assert artifact["sha256"] == comparator.self_hash(artifact)


@pytest.mark.parametrize("channel", ["model", "audio"])
def test_typed_only_semantic_failure_blocks(channel: str, tmp_path: Path) -> None:
    kwargs = {"speech_model" if channel == "model" else "speech_audio": (True,) * 4}
    comparator, typed, speech, runtime = prepare_pair(tmp_path, **kwargs)

    artifact = comparator.compare(typed, speech, runtime)

    assert artifact["passed"] is False
    assert len(artifact["blocking_reasons"]) == 4
    assert all("typed_only" in reason for reason in artifact["blocking_reasons"])


@pytest.mark.parametrize("channel", ["model", "audio"])
def test_speech_only_semantic_failure_is_disclosed_but_nonblocking(
    channel: str, tmp_path: Path
) -> None:
    kwargs = {"typed_model" if channel == "model" else "typed_audio": (True,) * 4}
    comparator, typed, speech, runtime = prepare_pair(tmp_path, **kwargs)

    artifact = comparator.compare(typed, speech, runtime)

    assert artifact["passed"] is True
    field = "model_text_semantic" if channel == "model" else "response_audio_semantic"
    assert all(case[field]["speech_only_failure"] for case in artifact["comparisons"])


def test_comparator_rejects_asr_provenance_and_aggregate_mutations(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["external_asr"]["model_id"] = "wrong"
    speech.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        comparator.compare(typed, speech, runtime)

    comparator, typed, speech, runtime = prepare_pair(tmp_path / "aggregate")
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["external_asr"]["passing_speech_responses"] = 3
    speech.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate"):
        comparator.compare(typed, speech, runtime)


def test_comparator_rejects_identity_and_audio_hash_mutations(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["experiment_sha256"] = "0" * 64
    speech.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        comparator.compare(typed, speech, runtime)

    comparator, typed, speech, runtime = prepare_pair(tmp_path / "audio")
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["responses"][0]["external_asr"]["source_audio_sha256"] = "0" * 64
    speech.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity"):
        comparator.compare(typed, speech, runtime)


def test_comparator_holds_promotion_on_thin_response_wer_margin(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    transcriber = load_qualification_module("transcribe_sustained")
    report = json.loads(typed.read_text(encoding="utf-8"))
    response = report["responses"][0]
    response["text"] = "The fresh word is ready now."
    transcript = "river wrong wrong is ready now."
    evaluation = transcriber.response_asr_evaluation(
        hypothesis=transcript,
        reference=response["text"],
        expected=response["expected_response_any"],
        excluded=[],
        max_wer=0.5,
        parity_arm=True,
    )
    assert evaluation["text_channel_wer"] == 0.5
    response["external_asr"].update({"transcript": transcript, **evaluation})
    raw = tmp_path / "runtime/typed/report.json"
    raw_report = json.loads(raw.read_text(encoding="utf-8"))
    raw_report["responses"][0]["text"] = response["text"]
    raw.write_text(json.dumps(raw_report), encoding="utf-8")
    report["external_asr_source_report"]["sha256"] = sha256_file(raw)
    typed.write_text(json.dumps(report), encoding="utf-8")

    artifact = comparator.compare(typed, speech, runtime)

    assert artifact["passed"] is False
    assert artifact["blocking_reasons"] == ["tool-freshness-1:response_asr_wer_margin_below_0.1"]


def v2_disposition(
    comparator,
    *,
    margin: float,
    typed_text: str = "The fresh word is ready now",
    speech_text: str = "The fresh word is ready now",
    model_shared_failure: bool = True,
    audio_shared_failure: bool = True,
) -> dict:
    return comparator._margin_v2_disposition(
        typed_case={
            "model_text": typed_text,
            "response_asr": {"audio_integrity_passed": True},
        },
        speech_case={
            "model_text": speech_text,
            "response_asr": {"audio_integrity_passed": True},
        },
        comparison={
            "case_id": "tool-freshness-1",
            "response_asr_wer": {"minimum_margin": margin},
            "model_text_semantic": {"shared_failure": model_shared_failure},
            "response_audio_semantic": {"shared_failure": audio_shared_failure},
        },
    )


@pytest.mark.parametrize(
    ("margin", "expected"),
    [(0.1, "clear"), (0.0999, "review_hold"), (0.0, "review_hold")],
)
def test_parity_margin_v2_boundary_is_prospective_and_exact(margin: float, expected: str) -> None:
    comparator = load_qualification_module("compare_tool_freshness_modalities")
    assert v2_disposition(comparator, margin=margin)["disposition"] == expected


@pytest.mark.parametrize(
    "mutation",
    [
        {"model_shared_failure": False},
        {"audio_shared_failure": False},
        {"speech_text": "Different model text"},
    ],
)
def test_parity_margin_v2_retains_hard_block_for_non_shared_or_unequal_cases(
    mutation: dict,
) -> None:
    comparator = load_qualification_module("compare_tool_freshness_modalities")
    result = v2_disposition(comparator, margin=0.0, **mutation)
    assert result["disposition"] == "hard_block"


def test_parity_margin_v2_degenerate_reference_is_not_an_independent_waiver() -> None:
    comparator = load_qualification_module("compare_tool_freshness_modalities")
    result = v2_disposition(
        comparator,
        margin=0.0,
        typed_text="fresh word",
        speech_text="fresh word",
        audio_shared_failure=False,
    )
    assert result["degenerate_reference"] is True
    assert result["disposition"] == "hard_block"


def test_parity_margin_v2_records_v1_and_requires_attended_review(
    tmp_path: Path,
) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    set_first_response_transcript(
        tmp_path,
        (typed, speech),
        text="The fresh word is ready now.",
        transcript="Wrong wrong wrong is ready now.",
    )

    strict = comparator.compare(typed, speech, runtime)
    artifact = comparator.compare_v2(typed, speech, runtime)

    assert strict["passed"] is False
    assert artifact["schema"] == 2
    assert artifact["kind"] == "tool_freshness_input_modality_parity_v2"
    assert artifact["core_passed"] is True
    assert artifact["passed"] is False
    assert artifact["attended_review_required"] is True
    assert artifact["unattended_promotion_eligible"] is False
    assert [item["case_id"] for item in artifact["review_holds"]] == ["tool-freshness-1"]
    assert artifact["verdicts"]["strict_margin_v1"] == {
        "blocking_reasons": strict["blocking_reasons"],
        "passed": False,
    }
    assert artifact["sha256"] == comparator.self_hash(artifact)


def test_parity_margin_v2_clear_margin_is_unattended_green(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    artifact = comparator.compare_v2(typed, speech, runtime)
    assert artifact["core_passed"] is True
    assert artifact["review_holds"] == []
    assert artifact["passed"] is True
    assert artifact["unattended_promotion_eligible"] is True


def test_parity_margin_v2_thin_positive_case_remains_hard_block(
    tmp_path: Path,
) -> None:
    comparator, typed, speech, runtime = prepare_pair(
        tmp_path,
        typed_model=(True, False, False, False),
        speech_model=(True, False, False, False),
        typed_audio=(True, False, False, False),
        speech_audio=(True, False, False, False),
    )
    set_first_response_transcript(
        tmp_path,
        (typed, speech),
        text="River is the fresh word ready",
        transcript="River wrong wrong wrong word ready",
    )
    artifact = comparator.compare_v2(typed, speech, runtime)
    assert artifact["core_passed"] is False
    assert artifact["review_holds"] == []
    assert artifact["margin_dispositions"][0]["disposition"] == "hard_block"
    assert artifact["passed"] is False


def test_parity_margin_v2_rejects_failed_absolute_integrity(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(typed.read_text(encoding="utf-8"))
    report["runtime_gate_passed"] = False
    report["passed"] = False
    typed.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="failed its independently recomputed gate"):
        comparator.compare_v2(typed, speech, runtime)


def test_parity_margin_v2_rejects_unmarked_historical_runtime_but_v1_accepts_it(
    tmp_path: Path,
) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(runtime.read_text(encoding="utf-8"))
    report.pop("comparison_policy")
    runtime.write_text(json.dumps(report), encoding="utf-8")
    assert comparator.compare(typed, speech, runtime)["passed"] is True
    with pytest.raises(ValueError, match="prospectively marked"):
        comparator.compare_v2(typed, speech, runtime)


def test_parity_margin_v2_policy_document_matches_executable_contract() -> None:
    comparator = load_qualification_module("compare_tool_freshness_modalities")
    policy = Path("docs/asr-judge-swap-policy.md").read_text(encoding="utf-8")
    assert f"`{comparator.PARITY_MARGIN_V2}`" in policy
    assert f"`WER <= {comparator.MAX_WER}`" in policy
    assert f"sub-`{comparator.MIN_WER_MARGIN}` margin" in policy
    assert "shortness never independently waives" in policy
    for evidence_hash in (
        "4889acc9305b691466ef80911280061a5d6eafd894a967c42d2a0d8381eae129",
        "4916ec737007d14d96058229392a98a960778adc80ff842214054cfedd26274b",
        "6cc3939709a28a87e4c20f8f8f090dfa61f3e971f92b94e8e503518512af25ac",
        "ccbbf4df38fa314569b2b202898f3bfe3caa04f572ba47aaa8d30744516c5a47",
        "1a9078fda74c78d623afbf9dd7bb31accc8819fc3a69b2b87617c4e0043b4a56",
        "e2920cb7b640904d7287ab5ab09a58c5110c3acb8e77a78ec4d62e5266444db0",
        "41e6bd0993046387349ffc90cc2590f720c05f921146c22da525b9efec4acee8",
        "4e3709270344f1a6fbc31fda69c6c4397b9c039cff47eec8912c5ccb88c9b111",
        "9714deaceb9d27dfddd34e46bc9c4c56a408f465bf52ca4cba700dbd13ccf6a2",
        "2e95857362d4e74c3bb7cd4b2d8edcdfecfc87fdf2bf90a913b078132d700f28",
    ):
        assert f"`{evidence_hash}`" in policy


@pytest.mark.parametrize(
    ("policy_args", "expected_schema", "expected_kind", "expected_policy"),
    [
        ([], 2, "tool_freshness_input_modality_parity_v2", "parity_margin_v2"),
        (
            ["--policy", "strict_margin_v1"],
            1,
            "tool_freshness_input_modality_parity_v1",
            None,
        ),
    ],
)
def test_comparator_cli_defaults_to_v2_and_retains_explicit_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_args: list[str],
    expected_schema: int,
    expected_kind: str,
    expected_policy: str | None,
) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    output = tmp_path / f"policy-{expected_schema}.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_tool_freshness_modalities.py",
            "--typed-report",
            str(typed),
            "--speech-report",
            str(speech),
            "--runtime-report",
            str(runtime),
            *policy_args,
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exited:
        comparator.main()
    assert exited.value.code == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema"] == expected_schema
    assert artifact["kind"] == expected_kind
    if expected_policy is None:
        assert "selected" not in artifact["policy"]
    else:
        assert artifact["policy"]["selected"] == expected_policy
    assert artifact["passed"] is True


def test_comparator_independently_rederives_response_semantics(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["responses"][0]["external_asr"]["semantic_match"] = True
    speech.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="verdict does not rederive"):
        comparator.compare(typed, speech, runtime)


def test_comparator_rederives_speech_input_wer_and_streamed_audio_length(
    tmp_path: Path,
) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["inputs"][0]["input_wer"] = 0.1
    speech.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="input WER"):
        comparator.compare(typed, speech, runtime)

    comparator, typed, speech, runtime = prepare_pair(tmp_path / "audio-length")
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["responses"][0]["audio_bytes"] -= 2
    speech.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="streamed audio bytes"):
        comparator.compare(typed, speech, runtime)


def test_comparator_requires_the_promoted_evaluator_image(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["external_asr"]["evaluator_image_id"] = f"sha256:{'d' * 64}"
    speech.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="promoted immutable"):
        comparator.compare(typed, speech, runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    [("torch_version", "wrong"), ("cuda_version", "13.0"), ("extra", "unbound")],
)
def test_comparator_requires_exact_platform_provenance_schema(
    field: str, value: str, tmp_path: Path
) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["external_asr"][field] = value
    speech.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        comparator.compare(typed, speech, runtime)


def test_comparator_rejects_same_length_canonical_pcm_substitution(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    response = report["responses"][0]
    canonical = Path(response["external_asr"]["wav_16k"])
    write_tone(canonical, 16_000)
    response["external_asr"]["canonical_wav_sha256"] = sha256_file(canonical)
    speech.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="does not derive"):
        comparator.compare(typed, speech, runtime)


def test_comparator_binds_arms_to_one_runtime_report_and_order(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(runtime.read_text(encoding="utf-8"))
    report["execution_order"] = ["speech", "typed"]
    runtime.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="bind both arms"):
        comparator.compare(typed, speech, runtime)

    comparator, typed, speech, runtime = prepare_pair(tmp_path / "nonpromotable")
    report = json.loads(runtime.read_text(encoding="utf-8"))
    report["promotion_eligible"] = False
    runtime.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="bind both arms"):
        comparator.compare(typed, speech, runtime)

    comparator, typed, speech, runtime = prepare_pair(tmp_path / "mixed")
    report = json.loads(runtime.read_text(encoding="utf-8"))
    alternate = tmp_path / "mixed/runtime/typed/alternate.json"
    alternate.write_text(
        (tmp_path / "mixed/runtime/typed/report.json").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    report["arms"]["typed"]["report"] = str(alternate)
    runtime.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        comparator.compare(typed, speech, runtime)


def test_comparator_rejects_evaluated_base_drift_from_raw_arm(tmp_path: Path) -> None:
    comparator, typed, speech, runtime = prepare_pair(tmp_path)
    report = json.loads(speech.read_text(encoding="utf-8"))
    report["fixture_manifest_path"] = "/tampered/after-asr.json"
    speech.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="base differs"):
        comparator.compare(typed, speech, runtime)
