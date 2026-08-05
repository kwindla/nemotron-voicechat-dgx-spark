from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron_voicechat_runtime.artifacts import Layout, load_config
from nemotron_voicechat_runtime.cli import (
    _owned_pipecat_process,
    _process_start_time,
    _wait_ports_available,
    model_container_command,
    parser,
)
from nemotron_voicechat_runtime.provenance import PRODUCTION_ENVIRONMENT


def test_checked_in_config_materializes_frozen_runtime_contract() -> None:
    config = load_config()
    environment = config["runtime"]["environment"]
    assert all(environment[name] == value for name, value in PRODUCTION_ENVIRONMENT.items())
    assert config["artifacts"]["release"]["repository"] == (
        "pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark"
    )
    affinity = config["runtime"]["cpu_affinity"]
    assert set(affinity["codec"]).isdisjoint(affinity["pocket_tts"])


def test_model_command_is_loopback_only_and_uses_signed_release(tmp_path: Path) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    command = model_container_command(config, layout, 9876)
    rendered = " ".join(command)
    assert "127.0.0.1:9876:8786" in command
    assert "0.0.0.0:9876:8786" not in rendered
    assert "/models/derived/manifests/release.json" in command
    assert "/models/derived/nano" in command
    assert "/models/derived/eartts" in command
    assert "HF_HOME=/models/huggingface" in command
    assert "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub" in command
    assert "HF_MODULES_CACHE=/tmp/voicechat-hf-modules" in command
    assert f"{layout.hf_home}:/models/huggingface:ro" in command
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in rendered
    assert "API_KEY" not in rendered


def test_cli_exposes_stable_foreground_commands() -> None:
    cli = parser()
    assert cli.parse_args(["bootstrap", "--offline"]).offline is True
    assert cli.parse_args(["up", "--host", "0.0.0.0", "--port", "9000"]).port == 9000
    assert cli.parse_args(["test", "--live"]).live is True
    assert cli.parse_args(["down"]).command == "down"
    assert cli.parse_args(["status"]).command == "status"


def test_stage1_cli_withholds_conversion_and_release_commands(capsys) -> None:
    cli = parser()
    for arguments in (
        ["release", "stage"],
        ["bootstrap", "--convert-from-source"],
        ["bootstrap", "--asr-model", "/tmp/parakeet.nemo"],
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(arguments)
        assert "error:" in capsys.readouterr().err


def test_no_persisted_auth_or_dotenv_contract_in_config() -> None:
    config_text = Path("config/production-candidate-1.toml").read_text(encoding="utf-8")
    assert "API_KEY" not in config_text
    assert "dotenv" not in config_text
    assert "EnvironmentFile" not in config_text
    config = load_config()
    assert json.dumps(config).find("password") == -1


def test_orphan_cleanup_validates_pid_identity_before_signaling(tmp_path: Path) -> None:
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"python\0-m\0nemotron_voicechat_pipecat.demo\0-t\0webrtc\0")
    (process / "stat").write_text(" ".join(["0"] * 21 + ["12345"]), encoding="utf-8")
    assert _owned_pipecat_process(123, tmp_path)
    assert _owned_pipecat_process(123, tmp_path, expected_start_time="12345")
    assert not _owned_pipecat_process(123, tmp_path, expected_start_time="other")
    assert _process_start_time(123, tmp_path) == "12345"
    (process / "cmdline").write_bytes(b"python\0unrelated.py\0")
    assert not _owned_pipecat_process(123, tmp_path)


def test_shutdown_waits_until_ports_can_be_rebound(monkeypatch) -> None:
    attempts = iter([False, False, True, True])
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._port_available",
        lambda _host, _port: next(attempts),
    )
    monkeypatch.setattr("nemotron_voicechat_runtime.cli.time.sleep", lambda _seconds: None)

    _wait_ports_available([("127.0.0.1", 8786), ("127.0.0.1", 7860)])


def test_port_probe_uses_server_rebind_semantics(monkeypatch) -> None:
    import socket

    from nemotron_voicechat_runtime.cli import _port_available

    calls = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def setsockopt(self, *args):
            calls.append(("setsockopt", args))

        def bind(self, endpoint):
            calls.append(("bind", endpoint))

    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli.socket.socket", lambda *_args: FakeSocket()
    )

    assert _port_available("127.0.0.1", 8786)
    assert calls == [
        ("setsockopt", (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)),
        ("bind", ("127.0.0.1", 8786)),
    ]


def test_port_probe_rejects_active_listener() -> None:
    import socket

    from nemotron_voicechat_runtime.cli import _port_available

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not _port_available("127.0.0.1", port)
