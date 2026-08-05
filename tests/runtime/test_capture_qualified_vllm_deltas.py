from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/provenance/capture-qualified-vllm-deltas.py"
SPEC = importlib.util.spec_from_file_location("capture_vllm_deltas", TOOL)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def put(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_capture_classifies_all_qualified_python_files(tmp_path: Path) -> None:
    qualified = tmp_path / "qualified"
    voicechat = tmp_path / "voicechat"
    native = tmp_path / "native"
    put(qualified, "both.py", "same\n")
    put(voicechat, "both.py", "same\n")
    put(native, "both.py", "same\n")
    put(qualified, "voicechat.py", "voicechat\n")
    put(voicechat, "voicechat.py", "voicechat\n")
    put(native, "voicechat.py", "native\n")
    put(qualified, "delta.py", "after\n")
    put(voicechat, "delta.py", "before\n")
    put(native, "delta.py", "a much longer native reference\n")
    put(qualified, "added.py", "new\n")

    index, patches = MODULE.capture(qualified, voicechat, native)

    assert index["qualified_python_files"] == 4
    assert index["counts"] == {
        "equal_both": 1,
        "equal_voicechat": 1,
        "equal_native": 0,
        "delta": 1,
        "added": 1,
    }
    assert set(patches) == {"patches/added.py.patch", "patches/delta.py.patch"}
    assert b"/dev/null" in patches["patches/added.py.patch"]


def test_capture_output_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    try:
        MODULE.write_capture(output, {}, {})
    except FileExistsError as error:
        assert str(output) in str(error)
    else:
        raise AssertionError("existing output was overwritten")
