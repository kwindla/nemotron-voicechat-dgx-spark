from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools/qualification"))

from verify_promotion_manifest import verify_promotion_manifest

MANIFEST = REPO_ROOT / "deploy/promotion-candidate/manifest.sha256"


def test_promotion_manifest_seals_every_production_launch_script() -> None:
    verified = verify_promotion_manifest(REPO_ROOT, MANIFEST)
    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "deploy/promotion-candidate").glob("start-production-*.sh")
    }
    assert expected <= set(verified)


def test_promotion_manifest_mutation_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "repo"
    target = copied / "deploy/promotion-candidate"
    for relative in verify_promotion_manifest(REPO_ROOT, MANIFEST):
        source_file = REPO_ROOT / relative
        target_file = copied / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
    shutil.copy2(MANIFEST, target / "manifest.sha256")
    launch = target / "start-production-notext-watchdog-hotfix-v1.sh"
    launch.write_bytes(launch.read_bytes() + b"\n# mutation\n")
    with pytest.raises(RuntimeError, match="promotion checksum drift"):
        verify_promotion_manifest(copied, target / "manifest.sha256")
