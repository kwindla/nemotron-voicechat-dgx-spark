from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from nemotron_voicechat_runtime.pocket_worker import EXPECTED_ASSETS
from nemotron_voicechat_runtime.semantic_corpus import corpus_sha256, load_semantic_corpus
from nemotron_voicechat_runtime.voice_function_fixture import (
    FIXTURE_CASE_ID,
    FIXTURE_INSTALLED_WORKER_SOURCE,
    FIXTURE_KIND,
    FIXTURE_LANGUAGE,
    FIXTURE_MANIFEST_NAME,
    FIXTURE_PCM_NAME,
    FIXTURE_PROJECT_SOURCE_PATHS,
    FIXTURE_SAMPLE_RATE,
    FIXTURE_SCHEMA,
    FIXTURE_SEED_SCHEME,
    FIXTURE_VOICE,
    FIXTURE_WAV_NAME,
    sha256_path,
    validate_fixture_source_provenance,
    validate_voice_function_fixture,
)
from nemotron_voicechat_runtime.voice_function_probe import (
    _MonotonicResponseFramePacer,
    _validate_trace_coverage,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tools/qualification/direct_text_semantic_corpus.json"
IMAGE_ID = "sha256:" + "a" * 64


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    pcm = (b"\x01\x00\xff\xff" * 800)
    pcm_path = tmp_path / FIXTURE_PCM_NAME
    wav_path = tmp_path / FIXTURE_WAV_NAME
    pcm_path.write_bytes(pcm)
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(FIXTURE_SAMPLE_RATE)
        output.writeframes(pcm)
    corpus = load_semantic_corpus(CORPUS)
    case = next(case for case in corpus["cases"] if case["id"] == FIXTURE_CASE_ID)
    assets = {
        name: {"bytes": details[0], "sha256": details[1]}
        for name, details in EXPECTED_ASSETS.items()
    }
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "kind": FIXTURE_KIND,
        "case_id": FIXTURE_CASE_ID,
        "corpus_sha256": corpus_sha256(CORPUS),
        "prompt": case["prompt"],
        "pcm": {
            "path": FIXTURE_PCM_NAME,
            "sample_rate": FIXTURE_SAMPLE_RATE,
            "sample_count": len(pcm) // 2,
            "bytes": len(pcm),
            "sha256": _digest(pcm),
        },
        "wav": {
            "path": FIXTURE_WAV_NAME,
            "bytes": wav_path.stat().st_size,
            "sha256": _digest(wav_path.read_bytes()),
        },
        "synthesis": {
            "language": FIXTURE_LANGUAGE,
            "voice": FIXTURE_VOICE,
            "quantize": True,
            "threads": 4,
            "cpus": [7, 8, 9, 15],
            "base_seed": 0,
            "seed_scheme": FIXTURE_SEED_SCHEME,
            "package_version": "2.1.0",
            "assets": assets,
        },
        "provenance": {
            "runtime_image": "voicechat:test",
            "runtime_image_id": IMAGE_ID,
            "source_sha256": {"source.py": "f" * 64},
        },
    }
    (tmp_path / FIXTURE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_validate_voice_function_fixture_round_trip(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    actual, pcm = validate_voice_function_fixture(
        tmp_path, corpus_path=CORPUS, expected_runtime_image_id=IMAGE_ID
    )

    assert actual == manifest
    assert len(pcm) == manifest["pcm"]["bytes"]


@pytest.mark.parametrize("target", ["pcm", "wav"])
def test_validate_voice_function_fixture_rejects_tampered_audio(
    tmp_path: Path, target: str
) -> None:
    _fixture(tmp_path)
    path = tmp_path / (FIXTURE_PCM_NAME if target == "pcm" else FIXTURE_WAV_NAME)
    path.write_bytes(path.read_bytes() + b"\x00")

    with pytest.raises(RuntimeError, match="match"):
        validate_voice_function_fixture(
            tmp_path, corpus_path=CORPUS, expected_runtime_image_id=IMAGE_ID
        )


def test_validate_voice_function_fixture_rejects_wrong_image(tmp_path: Path) -> None:
    _fixture(tmp_path)

    with pytest.raises(RuntimeError, match="runtime image ID mismatch"):
        validate_voice_function_fixture(
            tmp_path,
            corpus_path=CORPUS,
            expected_runtime_image_id="sha256:" + "b" * 64,
        )


def test_fixture_source_provenance_rehashes_project_and_installed_worker(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path)
    installed = tmp_path / "installed-worker.py"
    installed.write_text("installed\n", encoding="utf-8")
    manifest["provenance"]["source_sha256"] = {
        **{name: sha256_path(ROOT / name) for name in FIXTURE_PROJECT_SOURCE_PATHS},
        FIXTURE_INSTALLED_WORKER_SOURCE: sha256_path(installed),
    }

    validate_fixture_source_provenance(
        manifest, project_root=ROOT, installed_worker=installed
    )

    manifest["provenance"]["source_sha256"][FIXTURE_PROJECT_SOURCE_PATHS[0]] = "0" * 64
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        validate_fixture_source_provenance(
            manifest, project_root=ROOT, installed_worker=installed
        )


def test_trace_coverage_accepts_background_gap() -> None:
    trace = [
        {"phase": "voice_input", "phase_position": 0, "model_frame": 3},
        {"phase": "voice_input", "phase_position": 1, "model_frame": 4},
    ]
    coverage = [
        {
            "phase": "voice_input",
            "status": "fresh_model_position",
            "model_frame": 3,
        },
        {
            "phase": "voice_input",
            "status": "fresh_model_position",
            "model_frame": 4,
        },
        {
            "phase": "post_bos",
            "status": "no_new_model_position",
            "reason": "fc_background_owned_advancement",
            "background_active": True,
        },
    ]

    _validate_trace_coverage(trace, coverage, enabled=True)


def test_trace_coverage_rejects_unjustified_gap() -> None:
    coverage = [
        {
            "phase": "post_bos",
            "status": "no_new_model_position",
            "reason": "fc_background_owned_advancement",
            "background_active": False,
        }
    ]

    with pytest.raises(RuntimeError, match="unjustified"):
        _validate_trace_coverage([], coverage, enabled=True)


def test_trace_coverage_rejects_pre_eou_background_gap() -> None:
    coverage = [
        {
            "phase": "voice_input",
            "status": "no_new_model_position",
            "reason": "fc_background_owned_advancement",
            "background_active": True,
        }
    ]

    with pytest.raises(RuntimeError, match="unjustified"):
        _validate_trace_coverage([], coverage, enabled=True)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 10.0
        self.sleep_requests: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_requests.append(seconds)
        self.now += seconds


def test_response_frame_pacer_mirrors_production_deadline_clock() -> None:
    clock = _FakeClock()
    pacer = _MonotonicResponseFramePacer(
        frame_seconds=0.08,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert list(pacer.positions(4)) == [0, 1, 2, 3]
    assert clock.sleep_requests == pytest.approx([0.08, 0.08, 0.08])
    assert clock.now == pytest.approx(10.24)
    assert pacer.observed_frames == 4
    assert pacer.paced_transitions == 3


def test_response_frame_pacer_gives_background_time_without_joining() -> None:
    clock = _FakeClock()
    background = {"active": True}

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.now >= 10.16:
            background["active"] = False

    pacer = _MonotonicResponseFramePacer(
        frame_seconds=0.08,
        monotonic=clock.monotonic,
        sleep=sleep,
    )
    observed: list[int] = []
    for position in pacer.positions(8):
        observed.append(position)
        if not background["active"]:
            break

    assert observed == [0, 1, 2]
    assert pacer.paced_transitions == 2
    assert clock.now == pytest.approx(10.16)


def test_response_frame_pacer_keeps_a_hung_background_bounded() -> None:
    clock = _FakeClock()
    pacer = _MonotonicResponseFramePacer(
        frame_seconds=0.08,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    observed = [position for position in pacer.positions(5)]

    assert observed == [0, 1, 2, 3, 4]
    assert pacer.observed_frames == 5
    assert pacer.paced_transitions == 4
    assert clock.now == pytest.approx(10.32)
