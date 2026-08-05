from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path("tools/runtime/generate_runtime_sbom.py")
    spec = importlib.util.spec_from_file_location("generate_runtime_sbom", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spdx_ids_are_stable_and_sanitized() -> None:
    module = load_module()
    assert module.spdx_id("pypi", "Some_Package", "1.0+cpu") == (
        "SPDXRef-pypi-Some-Package-1.0-cpu"
    )


def test_package_has_spdx_and_purl_identity() -> None:
    module = load_module()
    item = module.package("torch", "2.9.1+cpu", "pypi")
    assert item["SPDXID"] == "SPDXRef-pypi-torch-2.9.1-cpu"
    assert item["externalRefs"][0]["referenceLocator"] == "pkg:pypi/torch@2.9.1+cpu"
