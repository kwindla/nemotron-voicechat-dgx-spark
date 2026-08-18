"""Regression tests for the pinned Pipecat prebuilt UI repair.

The repair rewrites a vendored browser asset, so its safety rests entirely on
refusing to touch anything but the exact revision it was verified against.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nemotron_voicechat_runtime import pipecat_ui_patch as patch

ASSET_DIR = "lib/python3.12/site-packages/pipecat_ai_prebuilt/client/dist/assets"


def _install(venv: Path, name: str, body: str) -> Path:
    assets = venv / ASSET_DIR
    assets.mkdir(parents=True, exist_ok=True)
    asset = assets / name
    asset.write_text(body, encoding="utf-8")
    return asset


@pytest.fixture
def unpatched_body() -> str:
    """The real installed chunk, so the pinned hashes are exercised for real."""
    repo_venv = Path(__file__).resolve().parents[2] / ".venv"
    for candidate in patch.candidates(repo_venv):
        body = candidate.read_text(encoding="utf-8")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest == patch.UNPATCHED_SHA256:
            return body
        if digest == patch.PATCHED_SHA256:
            return body.replace(patch._REPAIRED, patch._ANCHOR)
    pytest.skip("pipecat prebuilt UI is not installed in the repo venv")


def test_repairs_the_pinned_revision_and_is_idempotent(tmp_path, unpatched_body):
    venv = tmp_path / "venv"
    asset = _install(venv, "index.module-BotpBYFo.js", unpatched_body)

    assert "repaired" in patch.apply(venv)
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == patch.PATCHED_SHA256
    assert patch._REPAIRED in asset.read_text(encoding="utf-8")

    # A second start must not double-apply or fail.
    assert patch.apply(venv) == "pipecat prebuilt UI already repaired"
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == patch.PATCHED_SHA256


def test_repair_settles_the_connect_promise(tmp_path, unpatched_body):
    """The defect is the missing resolver call; assert the repair adds it."""
    venv = tmp_path / "venv"
    asset = _install(venv, "index.module-BotpBYFo.js", unpatched_body)
    assert patch._ANCHOR in unpatched_body

    patch.apply(venv)

    repaired = asset.read_text(encoding="utf-8")
    assert patch._ANCHOR not in repaired
    assert "this._connectResolve(),this._connectResolve=null" in repaired


def test_ignores_sibling_chunks(tmp_path, unpatched_body):
    """Only the chunk carrying DailyMediaManager may be rewritten."""
    venv = tmp_path / "venv"
    target = _install(venv, "index.module-BotpBYFo.js", unpatched_body)
    sibling = _install(venv, "index.module-DepSAiaV.js", "export const other = 1;\n")
    before = sibling.read_bytes()

    patch.apply(venv)

    assert sibling.read_bytes() == before
    assert hashlib.sha256(target.read_bytes()).hexdigest() == patch.PATCHED_SHA256


def test_fails_closed_on_an_unrecognised_revision(tmp_path):
    venv = tmp_path / "venv"
    _install(venv, "index.module-DEADBEEF.js", "a different upstream build")

    with pytest.raises(patch.PipecatUIPatchError) as excinfo:
        patch.apply(venv)
    assert "pinned" in str(excinfo.value)


def test_absent_install_is_not_an_error(tmp_path):
    assert "not installed" in patch.apply(tmp_path / "venv")
