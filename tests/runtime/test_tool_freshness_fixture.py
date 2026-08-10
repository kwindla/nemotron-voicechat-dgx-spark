from __future__ import annotations

import importlib.util
import json
import sys
import wave
from copy import deepcopy
from pathlib import Path

import pytest

from nemotron_voicechat_runtime.pocket_worker import EXPECTED_ASSETS
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    FIXTURE_INSTALLED_WORKER_SOURCE,
    FIXTURE_KIND,
    FIXTURE_KIND_V1,
    FIXTURE_PROJECT_SOURCE_PATHS,
    FIXTURE_SAMPLE_RATE,
    FIXTURE_SCHEMA,
    FIXTURE_SCHEMA_V1,
    SCENARIO_SHA256_V1,
    SCENARIO_SHA256_V2,
    TOOL_ONLY_PROMPTS,
    TOOL_ONLY_PROMPTS_V1,
    TOOL_RESULT_ORDINALS,
    TOOL_VERIFICATION_WORDS,
    canonical_sha256,
    self_hash,
    sha256_bytes,
    sha256_path,
    validate_tool_freshness_fixtures,
)

IMAGE_ID = f"sha256:{'a' * 64}"


def load_generator():
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        path = Path("tools/qualification/generate_tool_freshness_fixtures.py")
        spec = importlib.util.spec_from_file_location("generate_tool_freshness_fixtures", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def build_fixture(tmp_path: Path) -> tuple[dict, str]:
    scenario_sha256 = SCENARIO_SHA256_V2
    fixtures = []
    for ordinal, prompt in enumerate(TOOL_ONLY_PROMPTS, 1):
        case_id = f"tool-freshness-{ordinal}"
        pcm = bytes([ordinal, 0]) * 100
        pcm_path = tmp_path / f"{case_id}.pcm"
        wav_path = tmp_path / f"{case_id}.wav"
        pcm_path.write_bytes(pcm)
        with wave.open(str(wav_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(FIXTURE_SAMPLE_RATE)
            output.writeframes(pcm)
        fixtures.append(
            {
                "case_id": case_id,
                "ordinal": ordinal,
                "text": prompt,
                "pcm": {
                    "path": pcm_path.name,
                    "sample_rate": FIXTURE_SAMPLE_RATE,
                    "sample_count": len(pcm) // 2,
                    "bytes": len(pcm),
                    "sha256": sha256_bytes(pcm),
                },
                "wav": {
                    "path": wav_path.name,
                    "sample_rate": FIXTURE_SAMPLE_RATE,
                    "channels": 1,
                    "sample_width": 2,
                    "samples": len(pcm) // 2,
                    "bytes": wav_path.stat().st_size,
                    "sha256": sha256_path(wav_path),
                },
            }
        )
    sources = {
        name: "c" * 64 for name in (*FIXTURE_PROJECT_SOURCE_PATHS, FIXTURE_INSTALLED_WORKER_SOURCE)
    }
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "kind": FIXTURE_KIND,
        "scenario_sha256": scenario_sha256,
        "fixtures": fixtures,
        "synthesis": {
            "language": "english_2026-04",
            "voice": "alba",
            "quantize": True,
            "threads": 4,
            "cpus": [7, 8, 9, 15],
            "base_seed": 0,
            "seed_scheme": "sha256-text-plus-base-v1",
            "package_version": "2.1.0",
            "assets": {
                name: {"bytes": details[0], "sha256": details[1]}
                for name, details in EXPECTED_ASSETS.items()
            },
        },
        "provenance": {
            "runtime_image": "fixture-image",
            "runtime_image_id": IMAGE_ID,
            "source_sha256": sources,
        },
    }
    manifest["sha256"] = self_hash(manifest)
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest, scenario_sha256


def write_manifest(path: Path, manifest: dict, *, rehash: bool = True) -> None:
    if rehash:
        manifest["sha256"] = self_hash(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_scenario_uses_one_prompt_and_tool_fixture_source() -> None:
    generator = load_generator()
    from nemotron_voicechat_runtime import tool_freshness_contract

    scenario = generator.scenario_document()

    assert scenario["prompts"] == list(TOOL_ONLY_PROMPTS)
    assert generator.scenario_document is tool_freshness_contract.scenario_document
    assert [item["verification_word"] for item in scenario["tool_outputs"]] == list(
        TOOL_VERIFICATION_WORDS
    )
    assert [item["ordinal_word"] for item in scenario["tool_outputs"]] == list(TOOL_RESULT_ORDINALS)
    assert [item["excluded_responses"] for item in scenario["tool_outputs"]] == [
        [],
        ["river"],
        ["river", "candle"],
        ["river", "candle", "garden"],
    ]
    assert all(item["semantic_required"] is True for item in scenario["tool_outputs"])
    assert canonical_sha256(scenario) == SCENARIO_SHA256_V2
    assert canonical_sha256(generator.scenario_document()) == SCENARIO_SHA256_V2


def test_v1_and_v2_scenarios_are_exactly_versioned() -> None:
    from nemotron_voicechat_runtime.tool_freshness_contract import (
        scenario_document_for_sha256,
        scenario_document_v1,
        scenario_document_v2,
    )

    assert canonical_sha256(scenario_document_v1()) == SCENARIO_SHA256_V1
    assert canonical_sha256(scenario_document_v2()) == SCENARIO_SHA256_V2
    assert scenario_document_for_sha256(SCENARIO_SHA256_V1) == scenario_document_v1()
    assert scenario_document_for_sha256(SCENARIO_SHA256_V2) == scenario_document_v2()
    with pytest.raises(RuntimeError, match="unknown"):
        scenario_document_for_sha256("f" * 64)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["prompts"].__setitem__(0, "Call the UTC tool right now."),
        lambda value: value["prompts"].__setitem__(0, "Call tool number 7 right now."),
        lambda value: value["prompts"].__setitem__(
            0, "Call the tool for a fresh time result right now."
        ),
        lambda value: value["tool_outputs"][0]["function_output"].__setitem__(
            "spoken", "River. The fresh word is ready now."
        ),
        lambda value: value["tool_outputs"][0]["function_output"].__setitem__(
            "spoken", "River is ready now."
        ),
        lambda value: value["tool_outputs"][0]["function_output"].__setitem__(
            "identifier", "item seven"
        ),
    ],
)
def test_v2_measurability_lint_rejects_unscorable_corpus_drift(mutate) -> None:
    from nemotron_voicechat_runtime.tool_freshness_contract import (
        scenario_document_v2,
        validate_measurable_scenario_v2,
    )

    scenario = deepcopy(scenario_document_v2())
    mutate(scenario)
    with pytest.raises(RuntimeError):
        validate_measurable_scenario_v2(scenario)


def test_fixture_validator_binds_all_four_pcm_and_wav_carriers(tmp_path: Path) -> None:
    manifest, scenario_sha256 = build_fixture(tmp_path)

    validated, payloads = validate_tool_freshness_fixtures(
        tmp_path,
        expected_runtime_image_id=IMAGE_ID,
        expected_scenario_sha256=scenario_sha256,
    )

    assert validated == manifest
    assert len(payloads) == 4
    assert [item["text"] for item in validated["fixtures"]] == list(TOOL_ONLY_PROMPTS)


def test_archival_v1_fixture_validates_but_cross_version_mix_refuses(
    tmp_path: Path,
) -> None:
    manifest, _scenario_sha256 = build_fixture(tmp_path)
    manifest["schema"] = FIXTURE_SCHEMA_V1
    manifest["kind"] = FIXTURE_KIND_V1
    manifest["scenario_sha256"] = SCENARIO_SHA256_V1
    for fixture, prompt in zip(manifest["fixtures"], TOOL_ONLY_PROMPTS_V1, strict=True):
        fixture["text"] = prompt
    write_manifest(tmp_path / "manifest.json", manifest)

    validated, _payloads = validate_tool_freshness_fixtures(
        tmp_path,
        expected_runtime_image_id=IMAGE_ID,
        expected_scenario_sha256=SCENARIO_SHA256_V1,
    )
    assert validated["kind"] == FIXTURE_KIND_V1

    with pytest.raises(RuntimeError, match="identity|scenario|prompt"):
        validate_tool_freshness_fixtures(
            tmp_path,
            expected_runtime_image_id=IMAGE_ID,
            expected_scenario_sha256=SCENARIO_SHA256_V2,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["fixtures"][0].__setitem__("text", "changed"), "prompt"),
        (
            lambda value: value["fixtures"][0]["pcm"].__setitem__("sha256", "d" * 64),
            "PCM hash",
        ),
        (
            lambda value: value["fixtures"][0]["wav"].__setitem__("sha256", "d" * 64),
            "WAV does not match",
        ),
        (
            lambda value: value["synthesis"].__setitem__("base_seed", 1),
            "synthesis contract",
        ),
        (
            lambda value: value["provenance"].__setitem__("runtime_image_id", f"sha256:{'d' * 64}"),
            "runtime image provenance",
        ),
    ],
)
def test_fixture_validator_rejects_bound_field_mutations(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest, scenario_sha256 = build_fixture(tmp_path)
    mutation(manifest)
    write_manifest(tmp_path / "manifest.json", manifest)

    with pytest.raises(RuntimeError, match=message):
        validate_tool_freshness_fixtures(
            tmp_path,
            expected_runtime_image_id=IMAGE_ID,
            expected_scenario_sha256=scenario_sha256,
        )


def test_fixture_validator_rejects_self_hash_mutation(tmp_path: Path) -> None:
    manifest, scenario_sha256 = build_fixture(tmp_path)
    manifest["fixtures"][0]["ordinal"] = 9
    write_manifest(tmp_path / "manifest.json", manifest, rehash=False)

    with pytest.raises(RuntimeError, match="self-hash"):
        validate_tool_freshness_fixtures(
            tmp_path,
            expected_runtime_image_id=IMAGE_ID,
            expected_scenario_sha256=scenario_sha256,
        )
