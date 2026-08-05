from __future__ import annotations

import json
from pathlib import Path

import pytest
from reproduce_release_artifacts import (
    _make_release_readable,
    _run_stage,
    _validate_stage_marker,
    _verify_expected_artifact,
)


def test_final_release_is_host_readable_without_changing_payload(tmp_path: Path) -> None:
    release = tmp_path / "release"
    nested = release / "nano"
    nested.mkdir(parents=True)
    payload = nested / "model.safetensors"
    payload.write_bytes(b"qualified")
    release.chmod(0o700)
    nested.chmod(0o700)
    payload.chmod(0o600)

    _make_release_readable(release)

    assert release.stat().st_mode & 0o777 == 0o755
    assert nested.stat().st_mode & 0o777 == 0o755
    assert payload.stat().st_mode & 0o777 == 0o644
    assert payload.read_bytes() == b"qualified"


def test_stage_resumes_only_when_recorded_outputs_still_match(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    calls = 0

    def action() -> None:
        nonlocal calls
        calls += 1
        output.mkdir()
        (output / "payload").write_bytes(b"qualified")

    _run_stage(tmp_path, "convert", [output], action)
    _run_stage(tmp_path, "convert", [output], action)
    assert calls == 1
    assert _validate_stage_marker(tmp_path, "convert") is True

    (output / "payload").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="no longer matches"):
        _run_stage(tmp_path, "convert", [output], action)


def test_stage_rejects_unvalidated_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "partial"
    output.mkdir()
    with pytest.raises(RuntimeError, match="unvalidated partial"):
        _run_stage(tmp_path, "convert", [output], lambda: None)


def test_expected_artifact_rejects_inventory_and_hash_drift(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_bytes(b"config")
    import hashlib

    digest = hashlib.sha256(b"config").hexdigest()
    expected = {
        "composite_sha256": "composite",
        "files": {"config.json": digest},
    }
    assert _verify_expected_artifact(tmp_path, expected, "test")["files"] == {"config.json": digest}
    (tmp_path / "extra").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="inventory differs"):
        _verify_expected_artifact(tmp_path, expected, "test")


def test_stage_marker_is_self_authenticating(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    _run_stage(
        tmp_path,
        "convert",
        [output],
        lambda: (output.mkdir(), (output / "payload").write_bytes(b"x")),
    )
    marker = tmp_path / "stages/convert.json"
    payload = json.loads(marker.read_text())
    payload["stage"] = "tampered"
    marker.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="corrupt stage marker"):
        _validate_stage_marker(tmp_path, "convert")
