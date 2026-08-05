from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch",
    reason="conversion tensor tests run in the public runtime container",
)
from nano_gptq_calibration import PUBLIC_REPOSITORY, PUBLIC_REVISION
from package_replay_corpus import build_release_manifest


def _capture(root: Path, request_id: str, marker: float) -> Path:
    root.mkdir()
    provenance = {
        "repository": PUBLIC_REPOSITORY,
        "revision": PUBLIC_REVISION,
        "model_sha256": "model",
        "nano_sha256": "nano",
        "derived_from_ea": False,
    }
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "public_nano_deterministic_replay",
                "checkpoint_provenance": provenance,
            }
        )
    )
    payload = root / "nano-call-000000.pt"
    torch.save(
        {
            "inputs": {
                "input_embeds": torch.full((1, 2, 4), marker, dtype=torch.bfloat16),
                "request_id": request_id,
            }
        },
        payload,
    )
    import hashlib

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "call_index": 0,
                "file": payload.name,
                "bytes": payload.stat().st_size,
                "sha256": digest,
                "decode_steps": 1,
                "request_id": request_id,
                "sequence_epoch": 0,
            }
        )
        + "\n"
    )
    return root


def test_release_manifest_is_path_independent_and_disjoint(tmp_path: Path) -> None:
    calibration = _capture(tmp_path / "host-a", "cal", 1.0)
    evaluation = _capture(tmp_path / "host-b", "eval", 2.0)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "nano_gptq_corpus_selection",
                "calibration": [
                    {
                        "root_index": 0,
                        "include_request_ids": ["cal"],
                        "exclude_request_ids": {},
                    }
                ],
                "evaluation": [
                    {
                        "root_index": 0,
                        "include_request_ids": ["eval"],
                        "exclude_request_ids": {},
                    }
                ],
            }
        )
    )
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "fixture.json").write_text('[{"text": "public fixture"}]\n')

    manifest = build_release_manifest([calibration], [evaluation], selection, fixtures)
    encoded = json.dumps(manifest, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert manifest["redistributable"] is True
    assert manifest["contains_raw_user_recordings"] is False
    assert manifest["selected"]["calibration"]["calls"] == 1
    assert manifest["selected"]["evaluation"]["calls"] == 1
    assert manifest["roots"][0]["release_path"] == "calibration/0"
    assert manifest["roots"][1]["release_path"] == "evaluation/0"


def test_release_manifest_restores_source_labels_after_transport_renames_roots(
    tmp_path: Path,
) -> None:
    calibration = _capture(tmp_path / "0", "cal", 1.0)
    evaluation_parent = tmp_path / "evaluation"
    evaluation_parent.mkdir()
    evaluation = _capture(evaluation_parent / "0", "eval", 2.0)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "nano_gptq_corpus_selection",
                "calibration": [
                    {
                        "root_index": 0,
                        "include_request_ids": ["cal"],
                        "exclude_request_ids": {},
                    }
                ],
                "evaluation": [
                    {
                        "root_index": 0,
                        "include_request_ids": ["eval"],
                        "exclude_request_ids": {},
                    }
                ],
            }
        )
    )
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "fixture.json").write_text("[]\n")

    manifest = build_release_manifest(
        [calibration],
        [evaluation],
        selection,
        fixtures,
        source_names={
            ("calibration", 0): "calibration-original",
            ("evaluation", 0): "evaluation-original",
        },
    )

    assert [root["source_name"] for root in manifest["roots"]] == [
        "calibration-original",
        "evaluation-original",
    ]
    with pytest.raises(ValueError, match="cover every split/root index"):
        build_release_manifest(
            [calibration],
            [evaluation],
            selection,
            fixtures,
            source_names={("calibration", 0): "calibration-original"},
        )


def test_release_manifest_rejects_corrupt_payload(tmp_path: Path) -> None:
    calibration = _capture(tmp_path / "calibration", "cal", 1.0)
    evaluation = _capture(tmp_path / "evaluation", "eval", 2.0)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "nano_gptq_corpus_selection",
                "calibration": [
                    {
                        "root_index": 0,
                        "include_request_ids": ["cal"],
                        "exclude_request_ids": {},
                    }
                ],
                "evaluation": [
                    {
                        "root_index": 0,
                        "include_request_ids": ["eval"],
                        "exclude_request_ids": {},
                    }
                ],
            }
        )
    )
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "fixture.json").write_text("[]\n")
    (calibration / "nano-call-000000.pt").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="truncated replay payload"):
        build_release_manifest([calibration], [evaluation], selection, fixtures)
