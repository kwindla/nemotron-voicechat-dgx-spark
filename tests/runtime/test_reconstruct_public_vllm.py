from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from reconstruct_public_vllm import reconstruct


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_tree(root: Path, content: bytes) -> str:
    (root / "vllm").mkdir(parents=True)
    (root / "vllm/example.py").write_bytes(content)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    before = b"answer = 1\n"
    after = b"answer = 2\n"
    native = tmp_path / "native"
    voicechat = tmp_path / "voicechat"
    native_revision = _git_tree(native, before)
    voicechat_revision = _git_tree(voicechat, before)
    escrow = tmp_path / "escrow"
    (escrow / "patches").mkdir(parents=True)
    patch = escrow / "patches/example.py.patch"
    patch.write_text(
        "--- a/vllm/example.py\n+++ b/vllm/example.py\n@@ -1,99 +1,99 @@\n"
        "-answer = 1\n+answer = 2\n"
    )
    index = {
        "qualified_python_files": 1,
        "qualified_tree_sha256": _sha(
            b"example.py\x00" + str(len(after)).encode() + b"\x00" + _sha(after).encode() + b"\n"
        ),
        "files": [
            {
                "bytes": len(after),
                "classification": "delta",
                "patch": "patches/example.py.patch",
                "patch_sha256": _sha(patch.read_bytes()),
                "path": "example.py",
                "reference": "voicechat",
                "reference_sha256": _sha(before),
                "sha256": _sha(after),
            }
        ],
    }
    index_path = escrow / "index.json"
    index_path.write_text(json.dumps(index))
    bom = {
        "public_sources": {
            "native_vllm": {"revision": native_revision},
            "voicechat_vllm": {"revision": voicechat_revision},
        },
        "qualified_runtime_reference": {
            "delta_escrow": {"index_sha256": _sha(index_path.read_bytes())}
        },
    }
    bom_path = tmp_path / "bom.json"
    bom_path.write_text(json.dumps(bom))
    return native, voicechat, bom_path, index_path


def test_reconstructs_and_hashes_public_tree(tmp_path: Path) -> None:
    native, voicechat, bom, index = _fixture(tmp_path)
    output = tmp_path / "out/vllm"
    result = reconstruct(
        native_root=native,
        voicechat_root=voicechat,
        output=output,
        bom_path=bom,
        index_path=index,
    )
    assert (output / "example.py").read_text() == "answer = 2\n"
    assert result["files"] == 1


def test_refuses_source_drift_and_existing_output(tmp_path: Path) -> None:
    native, voicechat, bom, index = _fixture(tmp_path)
    (voicechat / "vllm/example.py").write_text("drift = True\n")
    with pytest.raises(ValueError, match="public source hash mismatch"):
        reconstruct(
            native_root=native,
            voicechat_root=voicechat,
            output=tmp_path / "out/vllm",
            bom_path=bom,
            index_path=index,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        reconstruct(
            native_root=native,
            voicechat_root=voicechat,
            output=existing,
            bom_path=bom,
            index_path=index,
        )
