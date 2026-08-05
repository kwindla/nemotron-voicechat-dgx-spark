from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def module():
    path = Path("tools/qualification/restart_cycle_gate.py")
    spec = importlib.util.spec_from_file_location("restart_cycle_gate", path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def test_launcher_command_uses_loopback_and_explicit_cache(tmp_path: Path) -> None:
    value = module()
    args = argparse.Namespace(
        cache_root=tmp_path / "cache",
        trace_root=tmp_path / "traces",
        model_port=9001,
        pipecat_port=9002,
    )
    command = value.launcher_command(args)
    assert command[-6:] == [
        "--model-port",
        "9001",
        "--host",
        "127.0.0.1",
        "--port",
        "9002",
    ]


def test_non_loopback_probe_refuses_vacuous_pass(monkeypatch) -> None:
    value = module()
    monkeypatch.setattr(value, "host_non_loopback_addresses", lambda: [])
    import pytest

    with pytest.raises(RuntimeError, match="no discoverable"):
        value.assert_model_not_exposed(8786)


def test_non_loopback_probe_treats_http_error_as_reachable(monkeypatch) -> None:
    import io
    import urllib.error

    import pytest

    value = module()
    monkeypatch.setattr(value, "host_non_loopback_addresses", lambda: ["192.0.2.10"])

    def reachable(*_args, **_kwargs):
        raise urllib.error.HTTPError("http://192.0.2.10", 404, "reachable", {}, io.BytesIO())

    monkeypatch.setattr(value.NO_PROXY_OPENER, "open", reachable)
    with pytest.raises(RuntimeError, match="raw model port is reachable"):
        value.assert_model_not_exposed(8786)
