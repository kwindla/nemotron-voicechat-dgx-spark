from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nemotron_voicechat_runtime import patch_step5_wedge_boundary_trace as patcher
from nemotron_voicechat_runtime.wedge_boundary_trace import descriptor_key


def _trace_probe(tmp_path: Path, *, enabled: bool, records: int) -> dict:
    code = f"""
import json
from nemotron_voicechat_runtime.wedge_boundary_trace import (
    close_trace, record_boundary, snapshot_records, trace_metadata,
)
for index in range({records}):
    record_boundary('B0', f'request-{{index}}', index=index)
result = trace_metadata()
result['snapshot'] = snapshot_records()
print(json.dumps(result, sort_keys=True))
close_trace()
"""
    env = dict(os.environ)
    env["NEMOTRON_WEDGE_TRACE"] = "1" if enabled else "0"
    env["NEMOTRON_WEDGE_TRACE_DIR"] = str(tmp_path)
    env["NEMOTRON_WEDGE_TRACE_CAPACITY"] = "10"
    env["NEMOTRON_WEDGE_TRACE_RUN_UUID"] = "unit-test"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _minimal_source(root: Path) -> None:
    for relative in patcher.TARGETS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n", encoding="utf-8")


@pytest.fixture
def atomic_patcher_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "qualified"
    source.mkdir()
    _minimal_source(source)
    hashes = {relative: patcher.sha256_file(source / relative) for relative in patcher.TARGETS}
    monkeypatch.setattr(patcher, "_qualified_hashes", lambda: hashes)
    digest = hashlib.sha256()
    for relative in sorted(hashes):
        path = source / relative
        digest.update(relative.encode() + b"\0")
        digest.update(str(path.stat().st_size).encode() + b"\0")
        digest.update(hashes[relative].encode() + b"\n")
    monkeypatch.setattr(patcher, "QUALIFIED_TREE_SHA256", digest.hexdigest())
    monkeypatch.setattr(
        patcher,
        "PATCHERS",
        {
            relative: (lambda text, relative=relative: text + f"# {relative}\n")
            for relative in patcher.TARGETS
        },
    )
    return source


def test_trace_is_off_by_default_and_inert(tmp_path: Path) -> None:
    metadata = _trace_probe(tmp_path, enabled=False, records=11)
    assert metadata == {"enabled": False, "snapshot": []}
    assert list(tmp_path.iterdir()) == []


def test_preallocated_sidecar_saturates_and_retains_v2_schema(tmp_path: Path) -> None:
    metadata = _trace_probe(tmp_path, enabled=True, records=11)
    assert metadata["capacity"] == 10
    assert metadata["records"] == 10
    assert metadata["dropped"] == 1
    assert metadata["schema_version"] == 2
    assert metadata["snapshot"][0]["task"] is None
    assert metadata["snapshot"][0]["campaign_uuid"] == "unit-test"
    sidecar = Path(metadata["path"])
    assert sidecar.stat().st_size == 10 * 512


def test_descriptor_digest_is_canonical() -> None:
    assert descriptor_key({"b": 2, "a": 1}) == descriptor_key({"a": 1, "b": 2})


def test_diagnostic_patcher_fails_closed_on_anchor_drift() -> None:
    assert patcher.replace_once("before anchor after", "anchor", "new", "fixture") == (
        "before new after"
    )
    with pytest.raises(ValueError, match="expected one anchor, found 0"):
        patcher.replace_once("drift", "anchor", "new", "fixture")


def test_patcher_publishes_copy_without_mutating_source(
    atomic_patcher_fixture: Path,
) -> None:
    source = atomic_patcher_fixture
    before = _tree_hash(source)
    output = source.parent / "published"
    patcher.apply_patch(source, output)
    assert _tree_hash(source) == before
    assert output.is_dir()
    assert (output / patcher.MANIFEST_NAME).is_file()


@pytest.mark.parametrize(
    "failure_point",
    [
        "after_verify",
        "after_copy",
        "after_target_0",
        "after_target_1",
        "after_target_2",
        "after_target_3",
        "before_compile",
        "after_compile",
        "before_publish",
    ],
)
def test_patcher_failure_points_leave_source_immutable_and_output_absent(
    atomic_patcher_fixture: Path,
    failure_point: str,
) -> None:
    source = atomic_patcher_fixture
    before = _tree_hash(source)
    output = source.parent / f"failed-{failure_point}"
    with pytest.raises(RuntimeError, match="injected patcher failure"):
        patcher.apply_patch(source, output, failure_point=failure_point)
    assert _tree_hash(source) == before
    assert not output.exists()


def test_patcher_rejects_alias_and_nesting(atomic_patcher_fixture: Path) -> None:
    source = atomic_patcher_fixture
    with pytest.raises(ValueError, match="distinct and unnested"):
        patcher.apply_patch(source, source)
    with pytest.raises(ValueError, match="distinct and unnested"):
        patcher.apply_patch(source, source / "nested")


def test_patcher_rejects_non_target_source_drift_before_writing(
    atomic_patcher_fixture: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = atomic_patcher_fixture
    extra = source / "v1/non_target.py"
    extra.write_text("qualified = True\n", encoding="utf-8")
    expected = {
        **patcher._qualified_hashes(),
        "v1/non_target.py": patcher.sha256_file(extra),
    }
    digest = hashlib.sha256()
    for relative in sorted(expected):
        path = source / relative
        digest.update(relative.encode() + b"\0")
        digest.update(str(path.stat().st_size).encode() + b"\0")
        digest.update(expected[relative].encode() + b"\n")
    monkeypatch.setattr(patcher, "_qualified_hashes", lambda: expected)
    monkeypatch.setattr(patcher, "QUALIFIED_TREE_SHA256", digest.hexdigest())
    extra.write_text("qualified = False\n", encoding="utf-8")
    output = source.parent / "drift-output"
    with pytest.raises(ValueError, match="qualified source hash mismatch"):
        patcher.apply_patch(source, output)
    assert not output.exists()
