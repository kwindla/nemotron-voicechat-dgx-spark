from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


def load_module():
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        path = Path("tools/qualification/compare_asr_judges.py")
        spec = importlib.util.spec_from_file_location("compare_asr_judges", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def write_wav(path: Path) -> None:
    samples = (np.sin(np.arange(1600) * 2 * np.pi * 220 / 16_000) * 4000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples.tobytes())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def reports(tmp_path: Path, module, *, semantic: bool = False):
    wav = tmp_path / "answer.wav"
    write_wav(wav)
    source_hash = module.sha256_file(wav)
    common = {
        "audio_path": str(wav),
        "text": "alpha beta gamma delta",
        "expected_response_any": ["alpha"] if semantic else ["cobalt"],
    }

    def response(transcript: str, wer: float, tokens=None):
        external = {
            "classification": "speech",
            "transcript": transcript,
            "text_channel_wer": wer,
            "expected_response_none": [],
            "semantic_match": semantic,
            "negative_semantic_match": True,
            "passed": semantic and wer <= 0.5,
        }
        if tokens is not None:
            external |= {
                "token_ids": tokens,
                "source_audio_sha256": source_hash,
                "canonical_wav_sha256": "c" * 64,
            }
        return common | {"external_asr": external}

    provenance = {"evaluator_image_id": f"sha256:{'a' * 64}", "decoder": "greedy"}
    old_gate = {
        "max_wer": 0.5,
        "response_count": 1,
        "speech_responses": 1,
        "passing_speech_responses": 0,
        "near_silent_responses": 0,
        "near_silent_rate": 0.0,
        "qualified_fp32_base_rate": 0.0,
        "passed": False,
    }
    new_gate = {
        **old_gate,
        "passing_speech_responses": int(semantic),
        "passed": semantic,
    }
    old = {
        "responses": [response("alpha", 0.75)],
        "external_asr": old_gate,
        "runtime_gate_passed": False,
        "passed": False,
    }
    new = {
        "responses": [response("alpha beta", 0.5, [1, 2, 3])],
        "external_asr": provenance | new_gate,
        "runtime_gate_passed": False,
        "passed": False,
    }
    rerun = json.loads(json.dumps(new))
    old_path, new_path, rerun_path = (
        tmp_path / "old.json",
        tmp_path / "new.json",
        tmp_path / "rerun.json",
    )
    write_json(old_path, old)
    write_json(new_path, new)
    write_json(rerun_path, rerun)
    write_wav(tmp_path / "response-001-16k.wav")
    model = tmp_path / "old.nemo"
    model.write_bytes(b"old model")
    old_provenance_dir = tmp_path / "old-provenance-rerun"
    old_provenance_dir.mkdir()
    old_provenance_report = old_provenance_dir / "report.json"
    write_json(old_provenance_report, old)
    retained_source_report = old_provenance_dir / "source-report.json"
    write_json(retained_source_report, old)
    write_wav(old_provenance_dir / "response-001-16k.wav")
    retained_decoder = old_provenance_dir / "legacy-transcribe-sustained.py"
    retained_decoder.write_bytes(
        subprocess.check_output(
            ("git", "show", "HEAD:tools/qualification/transcribe_sustained.py")
        )
    )
    docker_argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{old_provenance_dir.resolve()}:/work",
        "--volume",
        f"{model.resolve()}:/models/asr.nemo:ro",
        "--volume",
        (
            f"{retained_decoder.resolve()}:"
            "/workspace/project/tools/qualification/transcribe_sustained.py:ro"
        ),
        f"sha256:{'b' * 64}",
        "python",
        "/workspace/project/tools/qualification/transcribe_sustained.py",
        "--run-dir",
        "/work",
        "--model",
        "/models/asr.nemo",
        "--max-wer",
        "0.5",
        "--silence-dbfs",
        "-55",
        "--max-silent-rate",
        "0",
    ]
    old_provenance_log = old_provenance_dir / "execution.log"
    old_provenance_log.write_text(
        json.dumps(
            {
                "kind": "legacy_asr_execution",
                "schema": 2,
                "runtime_image_id": f"sha256:{'b' * 64}",
                "image_inspect_id": f"sha256:{'b' * 64}",
                "image_repo_digests": [],
                "model_mount": "/models/asr.nemo:ro",
                "source_run_dir": str(tmp_path.resolve()),
                "source_report_sha256": module.sha256_file(old_path),
                "retained_source_report": str(retained_source_report.resolve()),
                "retained_source_report_sha256": module.sha256_file(
                    retained_source_report
                ),
                "model_sha256": module.sha256_file(model),
                "container_model_sha256": module.sha256_file(model),
                "decoder_script_git_path": (
                    "HEAD:tools/qualification/transcribe_sustained.py"
                ),
                "decoder_script_git_blob": module.LEGACY_DECODER_GIT_BLOB,
                "decoder_script_sha256": module.LEGACY_DECODER_SHA256,
                "container_decoder_script_sha256": module.LEGACY_DECODER_SHA256,
                "retained_decoder_script": str(retained_decoder.resolve()),
                "network": "none",
                "gpu": "all",
                "ipc": "host",
                "docker_argv": docker_argv,
                "docker_argv_sha256": module.canonical_sha256(docker_argv),
            }
        )
        + '\nTranscribing: complete\n{"model": "/models/asr.nemo"}\n',
        encoding="utf-8",
    )
    old_positive_paths = []
    for label in ("eager", "graph"):
        lane = tmp_path / f"eartts-{label}"
        lane.mkdir()
        for name in (
            "eartts-component.wav",
            "eartts-component-16k.wav",
            "eartts-component-offline.wav",
            "eartts-component-offline-16k.wav",
        ):
            write_wav(lane / name)
        path = lane / "report.json"
        write_json(
            path,
            {
                "passed": True,
                "fixture": {"reference_text": "The answer is five."},
                "audio": {"path": "/work/eartts-component.wav"},
                "external_asr": {
                    "transcript": "The answer is five.",
                    "wer": 0.0,
                    "sha256": module.sha256_file(model),
                },
                "offline_diagnostic": {
                    "path": "/work/eartts-component-offline.wav",
                    "external_asr": {
                        "transcript": "The answer is five.",
                        "wer": 0.0,
                    },
                },
            },
        )
        old_positive_paths.append(path)
    new_positive_path = tmp_path / "new-positive.json"
    write_json(
        new_positive_path,
        {
            "passed": True,
            "external_asr": provenance,
            "cases": [
                {
                    "label": f"{lane}-{variant}",
                    "reference": "The answer is five.",
                    "transcript": "The answer is five.",
                    "wer": 0.0,
                    "passed": True,
                    "token_ids": [1],
                    "wav_bytes": (
                        tmp_path
                        / f"eartts-{lane}"
                        / f"eartts-component{'-offline' if variant == 'offline' else ''}-16k.wav"
                    ).stat().st_size,
                    "wav_sha256": module.sha256_file(
                        tmp_path
                        / f"eartts-{lane}"
                        / f"eartts-component{'-offline' if variant == 'offline' else ''}-16k.wav"
                    ),
                }
                for lane in ("eager", "graph")
                for variant in ("streaming", "offline")
            ],
        },
    )
    return (
        old_path,
        new_path,
        rerun_path,
        old_positive_paths,
        new_positive_path,
        model,
        old_provenance_report,
        old_provenance_log,
    )


def build(tmp_path: Path, module, *, policy: str, semantic: bool = False):
    (
        old,
        new,
        rerun,
        old_controls,
        new_controls,
        model,
        old_provenance_report,
        old_provenance_log,
    ) = reports(tmp_path, module, semantic=semantic)
    return module.build_calibration(
        [
            (
                "fixture",
                old,
                new,
                rerun,
                f"sha256:{'b' * 64}",
                1,
                old_provenance_report,
                old_provenance_log,
            )
        ],
        old_model_path=model,
        old_positive_paths=old_controls,
        new_positive_path=new_controls,
        policy=policy,
    )


def test_strict_policy_preserves_red_wer_flip_artifact(tmp_path: Path) -> None:
    module = load_module()
    report = build(tmp_path, module, policy="strict_component_v1")

    module.verify_self_hash(report)
    assert report["passed"] is False
    assert report["blocking_reasons"] == ["wer_threshold_outcome_flips"]
    assert report["derived"]["decision_flip_count"] == 0
    assert report["derived"]["wer_threshold_flip_count"] == 1
    assert report["corpus"][0]["wer_flip_is_non_load_bearing"] is True


def test_v2_accepts_disclosed_non_load_bearing_wer_flip(tmp_path: Path) -> None:
    module = load_module()
    report = build(tmp_path, module, policy="decision_concordance_v2")

    module.verify_self_hash(report)
    assert report["passed"] is True
    assert report["blocking_reasons"] == []


def test_v2_rejects_load_bearing_wer_flip(tmp_path: Path) -> None:
    module = load_module()
    report = build(
        tmp_path,
        module,
        policy="decision_concordance_v2",
        semantic=True,
    )

    assert report["passed"] is False
    assert report["blocking_reasons"] == [
        "decision_outcome_flips",
        "aggregate_verdict_flips",
        "load_bearing_wer_threshold_flips",
    ]


def test_self_hash_rejects_mutation(tmp_path: Path) -> None:
    module = load_module()
    report = build(tmp_path, module, policy="decision_concordance_v2")
    report["corpus"][0]["new"]["transcript"] = "mutated"

    with pytest.raises(ValueError, match="self-hash"):
        module.verify_self_hash(report)


def test_rerun_token_mismatch_fails_determinism_gate(tmp_path: Path) -> None:
    module = load_module()
    (
        old,
        new,
        rerun,
        old_controls,
        new_controls,
        model,
        old_provenance_report,
        old_provenance_log,
    ) = reports(tmp_path, module)
    rerun_report = json.loads(rerun.read_text(encoding="utf-8"))
    rerun_report["responses"][0]["external_asr"]["token_ids"] = [9]
    write_json(rerun, rerun_report)

    report = module.build_calibration(
        [
            (
                "fixture",
                old,
                new,
                rerun,
                f"sha256:{'b' * 64}",
                1,
                old_provenance_report,
                old_provenance_log,
            )
        ],
        old_model_path=model,
        old_positive_paths=old_controls,
        new_positive_path=new_controls,
        policy="decision_concordance_v2",
    )

    assert report["passed"] is False
    assert "new_evaluator_nondeterminism" in report["blocking_reasons"]


def test_near_silent_record_accepts_real_minimal_shape() -> None:
    module = load_module()

    record = module._asr_record(
        {
            "text": "",
            "external_asr": {"classification": "near_silent", "audio_dbfs": -90.0},
        },
        include_tokens=True,
    )

    assert record == {
        "classification": "near_silent",
        "audio_dbfs": -90.0,
        "transcript": None,
        "wer": None,
        "recomputed_wer": None,
        "wer_within_threshold": None,
        "semantic_match": None,
        "negative_semantic_match": None,
        "passed": False,
        "token_ids": [],
    }


def test_near_silent_pair_builds_end_to_end_and_binds_source(tmp_path: Path) -> None:
    module = load_module()
    fixture = list(reports(tmp_path, module))
    source_hash = module.sha256_file(tmp_path / "answer.wav")
    for index in (0, 1, 2, 6):
        path = fixture[index]
        report = json.loads(path.read_text(encoding="utf-8"))
        external = {"classification": "near_silent", "audio_dbfs": -120.0}
        if index in (1, 2):
            external["source_audio_sha256"] = source_hash
        report["responses"][0]["external_asr"] = external
        report["external_asr"] |= {
            "response_count": 1,
            "speech_responses": 0,
            "passing_speech_responses": 0,
            "near_silent_responses": 1,
            "near_silent_rate": 1.0,
            "qualified_fp32_base_rate": 1.0,
            "max_wer": 0.5,
            "passed": False,
        }
        write_json(path, report)
    retained_source_report = fixture[6].parent / "source-report.json"
    retained_source_report.write_bytes(fixture[0].read_bytes())
    log_lines = fixture[7].read_text(encoding="utf-8").splitlines()
    header = json.loads(log_lines[0])
    header["source_report_sha256"] = module.sha256_file(fixture[0])
    header["retained_source_report_sha256"] = module.sha256_file(
        retained_source_report
    )
    fixture[7].write_text(
        json.dumps(header) + "\n" + "\n".join(log_lines[1:]) + "\n",
        encoding="utf-8",
    )

    report = module.build_calibration(
        [
            (
                "fixture",
                fixture[0],
                fixture[1],
                fixture[2],
                f"sha256:{'b' * 64}",
                1,
                fixture[6],
                fixture[7],
            )
        ],
        old_model_path=fixture[5],
        old_positive_paths=fixture[3],
        new_positive_path=fixture[4],
        policy="decision_concordance_v2",
    )

    assert report["passed"] is True
    assert report["corpus"][0]["new"]["classification"] == "near_silent"
    assert report["corpus"][0]["wav"]["sha256"] == source_hash


def test_near_silent_pair_rejects_missing_source_hash(tmp_path: Path) -> None:
    module = load_module()
    fixture = list(reports(tmp_path, module))
    for path in (fixture[1], fixture[2]):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["responses"][0]["external_asr"] = {
            "classification": "near_silent",
            "audio_dbfs": -120.0,
        }
        write_json(path, report)

    with pytest.raises(ValueError, match="did not consume retained WAV bytes"):
        module.build_calibration(
            [
                (
                    "fixture",
                    fixture[0],
                    fixture[1],
                    fixture[2],
                    f"sha256:{'b' * 64}",
                    1,
                    fixture[6],
                    fixture[7],
                )
            ],
            old_model_path=fixture[5],
            old_positive_paths=fixture[3],
            new_positive_path=fixture[4],
            policy="decision_concordance_v2",
        )


def test_aggregate_pass_bits_are_rederived(tmp_path: Path) -> None:
    module = load_module()
    fixture = list(reports(tmp_path, module))
    new_path = fixture[1]
    new_report = json.loads(new_path.read_text(encoding="utf-8"))
    new_report["external_asr"]["passing_speech_responses"] = 1
    new_report["external_asr"]["passed"] = True
    write_json(new_path, new_report)

    with pytest.raises(ValueError, match="aggregate does not rederive"):
        module.build_calibration(
            [
                (
                    "fixture",
                    fixture[0],
                    fixture[1],
                    fixture[2],
                    f"sha256:{'b' * 64}",
                    1,
                    fixture[6],
                    fixture[7],
                )
            ],
            old_model_path=fixture[5],
            old_positive_paths=fixture[3],
            new_positive_path=fixture[4],
            policy="decision_concordance_v2",
        )


def test_gate_configuration_must_match_between_judges(tmp_path: Path) -> None:
    module = load_module()
    fixture = list(reports(tmp_path, module))
    new_path = fixture[1]
    rerun_path = fixture[2]
    for path in (new_path, rerun_path):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["external_asr"]["qualified_fp32_base_rate"] = 0.1
        write_json(path, report)

    report = module.build_calibration(
        [
            (
                "fixture",
                fixture[0],
                fixture[1],
                fixture[2],
                f"sha256:{'b' * 64}",
                1,
                fixture[6],
                fixture[7],
            )
        ],
        old_model_path=fixture[5],
        old_positive_paths=fixture[3],
        new_positive_path=fixture[4],
        policy="decision_concordance_v2",
    )

    assert report["passed"] is False
    assert report["derived"]["aggregate_verdicts_identical"] is False
    assert report["derived"]["aggregate_agreement"][0]["gate_config_identical"] is False


def test_rerun_contract_and_wav_identity_are_bound(tmp_path: Path) -> None:
    module = load_module()
    fixture = list(reports(tmp_path, module))
    rerun_path = fixture[2]
    rerun = json.loads(rerun_path.read_text(encoding="utf-8"))
    rerun["responses"][0]["text"] = "changed reference"
    write_json(rerun_path, rerun)

    with pytest.raises(ValueError, match="rerun reference text changed"):
        module.build_calibration(
            [
                (
                    "fixture",
                    fixture[0],
                    fixture[1],
                    fixture[2],
                    f"sha256:{'b' * 64}",
                    1,
                    fixture[6],
                    fixture[7],
                )
            ],
            old_model_path=fixture[5],
            old_positive_paths=fixture[3],
            new_positive_path=fixture[4],
            policy="decision_concordance_v2",
        )


def test_empty_corpus_is_rejected(tmp_path: Path) -> None:
    module = load_module()
    model = tmp_path / "old.nemo"
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="corpus is empty"):
        module.build_calibration(
            [],
            old_model_path=model,
            old_positive_paths=[],
            new_positive_path=tmp_path / "missing.json",
            policy="decision_concordance_v2",
        )


def test_positive_control_labels_must_be_exact_and_unique(tmp_path: Path) -> None:
    module = load_module()
    fixture = list(reports(tmp_path, module))
    controls = json.loads(fixture[4].read_text(encoding="utf-8"))
    controls["cases"][1]["label"] = controls["cases"][0]["label"]
    write_json(fixture[4], controls)

    with pytest.raises(ValueError, match="labels are duplicated"):
        module.build_calibration(
            [
                (
                    "fixture",
                    fixture[0],
                    fixture[1],
                    fixture[2],
                    f"sha256:{'b' * 64}",
                    1,
                    fixture[6],
                    fixture[7],
                )
            ],
            old_model_path=fixture[5],
            old_positive_paths=fixture[3],
            new_positive_path=fixture[4],
            policy="decision_concordance_v2",
        )


@pytest.mark.parametrize(
    ("name", "policy", "passed", "artifact_sha256"),
    [
        (
            "asr-judge-swap-calibration-v1.json",
            "strict_component_v1",
            False,
            "53c78175affac56cd3ad7a2b0433d2283e12ba7eee05435789198020c55d83b9",
        ),
        (
            "asr-judge-swap-calibration-v2.json",
            "decision_concordance_v2",
            True,
            "f4d6c4d8ff5d4810ce88b37328a6e53df13c8d9751f1ab603e3c0b5e4ee6dd4f",
        ),
    ],
)
def test_checked_in_calibration_artifacts_are_locked(
    name: str, policy: str, passed: bool, artifact_sha256: str
) -> None:
    module = load_module()
    artifact = json.loads((Path("docs/evidence") / name).read_text(encoding="utf-8"))

    module.verify_self_hash(artifact)
    assert artifact["sha256"] == artifact_sha256
    assert artifact["policy"] == policy
    assert artifact["passed"] is passed
    assert artifact["derived"]["response_count"] == 8
    assert artifact["derived"]["decision_flip_count"] == 0
    assert artifact["derived"]["wer_threshold_flip_count"] == 5
    assert artifact["derived"]["new_reverse_order_deterministic"] is True
    assert artifact["derived"]["positive_controls_passed"] is True
    assert artifact["derived"]["silence_control_passed"] is True
    assert len(artifact["positive_controls"]["old"]) == 4
    assert len(artifact["positive_controls"]["new"]["cases"]) == 4
    assert all(
        record["old_judge_rerun_attestation"]["passed"] is True
        for record in artifact["source_reports"]
    )
    assert artifact["judges"]["old"]["model_sha256"] == (
        "248e295b9f6390792e2254a5f94e0a5884204ce724ac3b4c4f1cb98f5e49730d"
    )
    candidate = artifact["judges"]["new"]
    assert candidate["evaluator_image_id"] == (
        "sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d"
    )
    assert candidate["backend_sha256"] == (
        "53b5142165fa8bfb1bff5685847157b7b37202d699e636de510ede8942f7d21a"
    )
    assert candidate["snapshot_sha256"] == (
        "cc5619455db64d897f460f874f99bc0b931f29b18830e2081a0369cc108c9a33"
    )
    assert candidate["decoder"] == {
        "attention_implementation": "sdpa",
        "num_lookahead_tokens": 13,
        "stopping": "encoder_exhaustion",
        "strategy": "greedy",
        "streaming": False,
    }
    assert candidate["deterministic_algorithms"] is True
    assert candidate["cuda_matmul_tf32"] is False
    assert candidate["cudnn_tf32"] is False
    observed_order = {
        record["label"]: record["rerun_corpus_order"]
        for record in artifact["source_reports"]
    }
    assert observed_order == {
        "r4": 2,
        "r5": 1,
    }
