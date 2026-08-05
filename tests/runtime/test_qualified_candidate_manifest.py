from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/provenance/verify-qualified-candidate.py"
SPEC = importlib.util.spec_from_file_location("qualified_candidate_verifier", TOOL)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def manifest() -> dict:
    return json.loads(
        (ROOT / "config/qualified-candidate-1.json").read_text(encoding="utf-8")
    )


def test_checked_in_candidate_manifest_is_valid() -> None:
    assert MODULE.validate_manifest(manifest()) == []


def test_opaque_image_is_comparison_only() -> None:
    payload = manifest()
    assert payload["qualified_runtime_reference"]["use"].startswith("comparison-only")
    assert payload["qualified_runtime_reference"]["opaque_imported_base_bytes"] > 0
    assert payload["derived_from_ea"] is False


def test_rejects_prohibited_public_build_input() -> None:
    payload = manifest()
    payload["public_sources"]["base_image"]["reference"] = (
        "nemotron-local/voicechat-vllm:ea-retired"
    )
    assert any(
        "prohibited public build input" in error
        for error in MODULE.validate_manifest(payload)
    )


def test_rejects_host_local_paths() -> None:
    payload = manifest()
    payload["public_sources"]["checkpoint"]["path"] = "/home/example/model"
    assert any("host-local path" in error for error in MODULE.validate_manifest(payload))


def test_rejects_malformed_file_sha() -> None:
    payload = manifest()
    payload["public_sources"]["checkpoint"]["files"]["config.json"] = "bad"
    assert any("invalid file SHA-256" in error for error in MODULE.validate_manifest(payload))


def test_rejects_weakened_reproducibility() -> None:
    payload = manifest()
    payload["qualified_artifacts"]["nano"]["reproducibility"]["runs"] = 1
    assert any("lacks dual byte-identical" in error for error in MODULE.validate_manifest(payload))


def test_rejects_opaque_image_as_build_input() -> None:
    payload = manifest()
    payload["qualified_runtime_reference"]["use"] = "release base"
    assert any("must be comparison-only" in error for error in MODULE.validate_manifest(payload))


def test_verify_tree_reports_missing_and_mismatched_files(tmp_path: Path) -> None:
    (tmp_path / "bad.bin").write_bytes(b"bad")
    errors = MODULE.verify_tree(
        tmp_path,
        {"missing.bin": "0" * 64, "bad.bin": "1" * 64},
        "fixture",
    )
    assert errors[0] == "fixture: missing missing.bin"
    assert "bad.bin SHA-256" in errors[1]


def test_verify_tree_reports_unreadable_files(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "unreadable.bin"
    path.write_bytes(b"payload")

    def fail(_path: Path) -> str:
        raise PermissionError("fixture denied")

    monkeypatch.setattr(MODULE, "sha256_file", fail)
    assert MODULE.verify_tree(tmp_path, {path.name: "0" * 64}, "fixture") == [
        "fixture: cannot read unreadable.bin: fixture denied"
    ]


def test_verify_tree_strict_mode_reports_unexpected_files(tmp_path: Path) -> None:
    (tmp_path / "expected.bin").write_bytes(b"expected")
    (tmp_path / "unexpected.bin").write_bytes(b"unexpected")
    errors = MODULE.verify_tree(
        tmp_path,
        {"expected.bin": MODULE.sha256_file(tmp_path / "expected.bin")},
        "fixture",
        strict=True,
    )
    assert errors == ["fixture: unexpected unexpected.bin"]
