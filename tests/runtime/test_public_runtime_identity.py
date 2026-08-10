from __future__ import annotations

import shutil
import stat
from pathlib import Path

import pytest

from nemotron_voicechat_runtime.public_runtime_identity import (
    PUBLIC_RUNTIME_PAYLOAD_LABEL,
    PUBLIC_RUNTIME_RECIPE_LABEL,
    PUBLIC_RUNTIME_SOURCE_LABEL,
    public_runtime_payload_sha256,
    public_runtime_recipe_sha256,
    public_runtime_recipe_snapshot,
    public_runtime_source_paths,
    public_runtime_source_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_identity_inventory_covers_installed_source_and_build_inputs(
    tmp_path: Path,
) -> None:
    paths = public_runtime_source_paths(REPO_ROOT)
    relatives = {path.relative_to(REPO_ROOT).as_posix() for path in paths}
    assert not any("__pycache__" in relative or relative.endswith(".pyc") for relative in relatives)
    assert "src/nemotron_voicechat_runtime/server.py" in relatives
    assert "src/nemotron_voicechat_pipecat/demo.py" in relatives
    assert (
        "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch"
    ) in relatives
    assert "container/Dockerfile.public-runtime" in relatives
    assert "container/Dockerfile.public-vllm" in relatives
    assert ".dockerignore" in relatives
    assert "pyproject.toml" in relatives
    assert "config/production-candidate-1.toml" in relatives
    assert "config/production-candidate-2.toml" in relatives

    for source in paths:
        destination = tmp_path / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    original = public_runtime_source_sha256(tmp_path)
    original_payload = public_runtime_payload_sha256(tmp_path)
    original_recipe = public_runtime_recipe_sha256(tmp_path)
    assert public_runtime_recipe_snapshot(tmp_path)

    ignored = tmp_path / "src/generated.egg-info/metadata"
    ignored.parent.mkdir()
    ignored.write_text("ignored", encoding="utf-8")
    pycache = tmp_path / "src/package/__pycache__/module.cpython-312.pyc"
    pycache.parent.mkdir(parents=True)
    pycache.write_bytes(b"ignored")
    assert public_runtime_source_sha256(tmp_path) == original

    ordinary = tmp_path / "src/package/untracked.py"
    ordinary.write_text("new runtime source\n", encoding="utf-8")
    assert public_runtime_payload_sha256(tmp_path) != original_payload
    assert public_runtime_source_sha256(tmp_path) != original
    ordinary.unlink()

    for relative in (
        "config/production-candidate-1.toml",
        "config/production-candidate-2.toml",
    ):
        policy = tmp_path / relative
        policy.write_bytes(policy.read_bytes() + b"\n# policy identity mutation\n")
        assert public_runtime_payload_sha256(tmp_path) != original_payload
        assert public_runtime_source_sha256(tmp_path) != original
        shutil.copy2(REPO_ROOT / relative, policy)

    dockerignore = tmp_path / ".dockerignore"
    dockerignore.write_bytes(dockerignore.read_bytes() + b"\nnew-pattern\n")
    assert public_runtime_recipe_sha256(tmp_path) != original_recipe
    assert public_runtime_source_sha256(tmp_path) != original
    shutil.copy2(REPO_ROOT / ".dockerignore", dockerignore)

    server = tmp_path / "src/nemotron_voicechat_runtime/server.py"
    server.write_bytes(server.read_bytes() + b"\n# source identity mutation\n")
    assert public_runtime_source_sha256(tmp_path) != original
    shutil.copy2(REPO_ROOT / "src/nemotron_voicechat_runtime/server.py", server)
    server.chmod(stat.S_IMODE(server.stat().st_mode) ^ stat.S_IXUSR)
    assert public_runtime_source_sha256(tmp_path) != original


@pytest.mark.parametrize("directory", [False, True])
def test_runtime_identity_rejects_nested_symlinks(tmp_path: Path, directory: bool) -> None:
    for source in public_runtime_source_paths(REPO_ROOT):
        destination = tmp_path / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    link = tmp_path / "src/untracked-link"
    target = (
        tmp_path / "src/nemotron_voicechat_runtime"
        if directory
        else tmp_path / "src/nemotron_voicechat_runtime/server.py"
    )
    link.symlink_to(target, target_is_directory=directory)

    with pytest.raises(RuntimeError, match="contains a symlink"):
        public_runtime_source_sha256(tmp_path)


def test_build_and_audit_bind_the_same_project_source_label() -> None:
    dockerfile = (REPO_ROOT / "container/Dockerfile.public-runtime").read_text()
    builder = (REPO_ROOT / "container/build-public-runtime.sh").read_text()
    audit = (REPO_ROOT / "container/audit-public-runtime.sh").read_text()

    assert "ARG VOICECHAT_SOURCE_SHA256" in dockerfile
    assert PUBLIC_RUNTIME_SOURCE_LABEL in dockerfile
    assert PUBLIC_RUNTIME_PAYLOAD_LABEL in dockerfile
    assert PUBLIC_RUNTIME_RECIPE_LABEL in dockerfile
    assert '--build-arg "VOICECHAT_SOURCE_SHA256=${project_source_sha256}"' in builder
    assert '--build-context "voicechat_recipe_snapshot=${recipe_snapshot_root}"' in builder
    assert "COPY --from=voicechat_recipe_snapshot" in dockerfile
    assert PUBLIC_RUNTIME_SOURCE_LABEL in audit
    assert "copied-source-identity.json" in audit
    assert (
        "COPY config/production-candidate-1.toml "
        "/opt/project/config/production-candidate-1.toml"
    ) in dockerfile
    assert (
        "COPY config/production-candidate-2.toml "
        "/opt/project/config/production-candidate-2.toml"
    ) in dockerfile
