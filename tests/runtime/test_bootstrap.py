from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nemotron_voicechat_runtime import bootstrap_download, cli
from nemotron_voicechat_runtime.artifacts import default_layout, is_lfs_pointer


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_release(root: Path) -> dict:
    payload = root / "nano/config.json"
    payload.parent.mkdir(parents=True)
    payload.write_text("{}", encoding="utf-8")
    files = [
        {
            "path": "nano/config.json",
            "bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        }
    ]
    manifest = {
        "schema": 1,
        "kind": "nemotron_voicechat_dgx_spark_hf_release",
        "repository": "pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark",
        "derived_from_ea": False,
        "files": files,
        "files_sha256": canonical_sha256(files),
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    path = root / "manifests/release.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def test_signed_release_verifier_rejects_any_payload_change(tmp_path: Path) -> None:
    manifest = make_release(tmp_path)
    result = bootstrap_download.verify_release(tmp_path, manifest["release_sha256"])
    assert result["release_sha256"] == manifest["release_sha256"]
    (tmp_path / "nano/config.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="byte count mismatch"):
        bootstrap_download.verify_release(tmp_path, manifest["release_sha256"])


def test_lfs_pointer_is_never_accepted_as_a_model(tmp_path: Path) -> None:
    pointer = tmp_path / "model.safetensors"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 10\n"
    )
    assert is_lfs_pointer(pointer)


def test_snapshot_download_is_atomic_and_scrubs_local_metadata(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "final"

    def fake_snapshot(_repository, **kwargs):
        local = Path(kwargs["local_dir"])
        (local / "config.json").write_text("{}")
        (local / ".cache/huggingface").mkdir(parents=True)
        (local / ".cache/huggingface/state").write_text("resume")

    monkeypatch.setattr(bootstrap_download, "snapshot_download", fake_snapshot)
    bootstrap_download._download_snapshot(
        repository="test/repo",
        revision="a" * 40,
        destination=destination,
        cache_dir=tmp_path / "hf",
        allow_patterns=["config.json"],
        offline=True,
    )
    assert (destination / "config.json").read_text() == "{}"
    assert not (destination / ".cache").exists()


def test_offline_manifest_lookup_is_explicitly_cache_only(tmp_path: Path, monkeypatch) -> None:
    release = make_release(tmp_path / "release")
    manifest_path = tmp_path / "release/manifests/release.json"
    observed = {}

    def fake_download(*_args, **kwargs):
        observed.update(kwargs)
        return str(manifest_path)

    monkeypatch.setattr(bootstrap_download, "hf_hub_download", fake_download)
    patterns, result = bootstrap_download._release_patterns(
        "test/repo", "b" * 40, tmp_path / "cache", True
    )
    assert observed["local_files_only"] is True
    assert result["release_sha256"] == release["release_sha256"]
    assert patterns == ["nano/config.json", "manifests/release.json"]


def test_artifact_stage_reports_start_completion_and_failure(capsys, monkeypatch) -> None:
    moments = iter([10.0, 12.5, 20.0, 21.25])
    monkeypatch.setattr(bootstrap_download.time, "monotonic", lambda: next(moments))

    with bootstrap_download._artifact_stage(2, 4, "verify parent checkpoint"):
        pass
    with pytest.raises(RuntimeError, match="broken"):
        with bootstrap_download._artifact_stage(3, 4, "verify tokenizer"):
            raise RuntimeError("broken")

    assert capsys.readouterr().out.splitlines() == [
        "[artifacts 2/4] verify parent checkpoint...",
        "[artifacts 2/4] complete (2.5s)",
        "[artifacts 3/4] verify tokenizer...",
        "[artifacts 3/4] failed after 1.2s",
    ]


def test_uv_offline_sync_uses_frozen_lock(tmp_path: Path, monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(cli, "_executable", lambda _name: "uv")
    monkeypatch.setattr(
        cli,
        "_run",
        lambda command, **kwargs: observed.append((command, kwargs["env"])),
    )
    layout = cli.Layout(tmp_path / "cache", tmp_path / "traces")
    cli._sync_host(layout, True)
    assert observed[0][0] == ["uv", "sync", "--frozen", "--offline"]
    assert observed[0][1]["UV_PROJECT_ENVIRONMENT"] == str(layout.cache / "venv")
    assert observed[0][1]["UV_CACHE_DIR"] == str(layout.cache / "uv")


def test_default_mutable_roots_are_outside_checkout(monkeypatch) -> None:
    monkeypatch.delenv("VOICECHAT_CACHE_ROOT", raising=False)
    monkeypatch.delenv("VOICECHAT_TRACE_ROOT", raising=False)
    layout = default_layout()
    assert not layout.cache.is_relative_to(cli.REPO_ROOT)
    assert not layout.traces.is_relative_to(cli.REPO_ROOT)


def test_public_build_supports_required_no_cache_gate() -> None:
    script = Path("container/build-public-runtime.sh").read_text(encoding="utf-8")
    assert "VOICECHAT_BUILD_NO_CACHE" in script
    assert "build_flags+=(--no-cache)" in script


def test_runtime_audit_uses_required_tool_and_fails_closed_on_scan_error() -> None:
    script = Path("container/audit-public-runtime.sh").read_text(encoding="utf-8")
    assert "\nrg " not in script
    assert "grep -E -i" in script
    assert "lineage_scan_status=$?" in script
    assert "lineage_scan_status} -ne 1" in script
    assert "Runtime image lineage scan failed" in script
