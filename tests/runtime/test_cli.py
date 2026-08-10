from __future__ import annotations

import hashlib
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron_voicechat_runtime.artifacts import Layout, PC2A_ENVIRONMENT, load_config
from nemotron_voicechat_runtime.cli import (
    _activate_converted_release,
    _bot_source_snapshot,
    _owned_launcher_process,
    _owned_pipecat_process,
    _process_start_time,
    _remove_stale_conversion_activations,
    _source_conversion_asr_model,
    _stop_stack,
    _tree_copy_gib,
    _wait_ports_available,
    command_release,
    command_restart_bot,
    command_up,
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


def test_pc2a_config_materializes_hotfix_runtime_contract() -> None:
    config = load_config(Path("config/production-candidate-2.toml"))
    assert config["candidate"] == "promotion-candidate-2a"
    assert config["image"]["runtime"] == (
        "nemotron-local/voicechat-vllm:promotion-candidate-2a"
    )
    environment = config["runtime"]["environment"]
    assert {name: environment.get(name) for name in PC2A_ENVIRONMENT} == PC2A_ENVIRONMENT
    assert environment["VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE"] == "0"


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


def test_model_command_enables_pad_pair_trace_only_when_requested(tmp_path: Path) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")

    ordinary = " ".join(model_container_command(config, layout, 9876))
    traced = model_container_command(config, layout, 9876, trace_pad_pair=True)

    assert "VOICECHAT_NANO_PAD_PAIR_TRACE" not in ordinary
    assert "VOICECHAT_NANO_PAD_PAIR_TRACE=1" in traced


def test_cli_exposes_stable_foreground_commands() -> None:
    cli = parser()
    assert cli.parse_args(["bootstrap", "--offline"]).offline is True
    conversion = cli.parse_args(
        ["bootstrap", "--convert-from-source", "--asr-model", "/tmp/parakeet.nemo"]
    )
    assert conversion.convert_from_source is True
    assert conversion.asr_model == "/tmp/parakeet.nemo"
    up = cli.parse_args(
        [
            "up",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload-bot",
            "--trace-pad-pair",
        ]
    )
    assert up.port == 9000
    assert up.reload_bot is True
    assert up.trace_pad_pair is True
    assert cli.parse_args(["restart-bot"]).timeout == 45.0
    assert cli.parse_args(["test", "--live"]).live is True
    assert cli.parse_args(["down"]).command == "down"
    assert cli.parse_args(["status"]).command == "status"
    assert cli.parse_args(["release", "stage"]).arguments == ["stage"]


def test_source_conversion_requires_a_regular_asr_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires --asr-model"):
        _source_conversion_asr_model(None)
    with pytest.raises(RuntimeError, match="not a regular file"):
        _source_conversion_asr_model(str(tmp_path / "missing.nemo"))
    checkpoint = tmp_path / "parakeet.nemo"
    checkpoint.write_bytes(b"checkpoint")
    assert _source_conversion_asr_model(str(checkpoint)) == checkpoint.resolve()


def test_source_conversion_transactionally_activates_verified_local_files(
    tmp_path: Path, monkeypatch
) -> None:
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    downloaded = layout.release
    (downloaded / "nano").mkdir(parents=True)
    (downloaded / "eartts").mkdir()
    (downloaded / "nano/model.safetensors").write_bytes(b"downloaded-nano")
    (downloaded / "eartts/model.safetensors").write_bytes(b"downloaded-eartts")
    (downloaded / "manifests").mkdir()
    (downloaded / "manifests/release.json").write_text("signed\n", encoding="utf-8")

    converted = tmp_path / "conversion/release"
    (converted / "nano").mkdir(parents=True)
    (converted / "eartts").mkdir()
    local_nano = converted / "nano/model.safetensors"
    local_eartts = converted / "eartts/model.safetensors"
    local_nano.write_bytes(b"source-nano")
    local_eartts.write_bytes(b"source-eartts")
    reproduction = {
        "schema": 1,
        "kind": "voicechat_release_artifact_reproduction",
    }
    reproduction_sha = hashlib.sha256(
        json.dumps(reproduction, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reproduction["sha256"] = reproduction_sha
    (converted / "reproduction.json").write_text(json.dumps(reproduction), encoding="utf-8")

    verified: list[Path] = []

    def verify(root: Path, expected_sha: str):
        assert expected_sha == "signed-release-sha"
        assert (root / "nano/model.safetensors").read_bytes() == b"source-nano"
        assert (root / "eartts/model.safetensors").read_bytes() == b"source-eartts"
        assert (root / "manifests/release.json").read_text() == "signed\n"
        verified.append(root)
        return {"release_sha256": expected_sha}

    monkeypatch.setattr("nemotron_voicechat_runtime.bootstrap_download.verify_release", verify)
    result = _activate_converted_release(layout, converted, "signed-release-sha")

    assert result == {
        "artifact_materialization": "source-converted",
        "reproduction_sha256": reproduction_sha,
    }
    assert len(verified) == 2
    assert (layout.release / "nano/model.safetensors").read_bytes() == local_nano.read_bytes()
    assert (layout.release / "eartts/model.safetensors").read_bytes() == local_eartts.read_bytes()
    assert not (layout.release / "nano/model.safetensors").samefile(local_nano)
    assert not (layout.release / "eartts/model.safetensors").samefile(local_eartts)
    assert not list(layout.artifacts.glob(".source-converted-release-*"))


def test_source_conversion_refuses_a_corrupt_reproduction_report(tmp_path: Path) -> None:
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    converted = tmp_path / "conversion/release"
    (converted / "nano").mkdir(parents=True)
    (converted / "eartts").mkdir()
    (converted / "reproduction.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "voicechat_release_artifact_reproduction",
                "sha256": "not-the-report-hash",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="failed its self-hash"):
        _activate_converted_release(layout, converted, "signed-release-sha")


def test_stale_source_conversion_activation_cleanup_is_fail_closed(tmp_path: Path) -> None:
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    stale = layout.artifacts / ".source-converted-release-interrupted"
    (stale / "downloaded-release").mkdir(parents=True)
    (stale / "downloaded-release/marker").write_text("old")

    _remove_stale_conversion_activations(layout)
    assert not stale.exists()

    unexpected = layout.artifacts / ".source-converted-release-not-a-directory"
    unexpected.write_text("unexpected")
    with pytest.raises(RuntimeError, match="unexpected source-conversion activation path"):
        _remove_stale_conversion_activations(layout)


def test_activation_copy_reservation_rounds_up_component_bytes(tmp_path: Path) -> None:
    nano = tmp_path / "nano"
    eartts = tmp_path / "eartts"
    nano.mkdir()
    eartts.mkdir()
    (nano / "weights").write_bytes(b"nano")
    (eartts / "weights").write_bytes(b"eartts")

    assert _tree_copy_gib((nano, eartts)) == 1
    assert _tree_copy_gib((tmp_path / "missing",)) == 0


def test_release_wrapper_strips_optional_argument_separator(
    tmp_path: Path, monkeypatch
) -> None:
    captured: list[list[str]] = []

    def run(command, **_kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("nemotron_voicechat_runtime.cli._run", run)
    args = parser().parse_args(
        ["--cache-root", str(tmp_path), "release", "--", "/tmp/stage", "--dry-run"]
    )
    assert command_release(args) == 0
    assert captured[0][-2:] == ["/tmp/stage", "--dry-run"]
    assert "--" not in captured[0]


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


def test_bot_restart_validates_launcher_identity(tmp_path: Path) -> None:
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(
        b"python\0-m\0nemotron_voicechat_runtime.cli\0up\0--reload-bot\0"
    )
    (process / "stat").write_text(" ".join(["0"] * 21 + ["12345"]), encoding="utf-8")
    assert _owned_launcher_process(123, tmp_path)
    assert _owned_launcher_process(123, tmp_path, expected_start_time="12345")
    assert not _owned_launcher_process(123, tmp_path, expected_start_time="other")
    (process / "cmdline").write_bytes(b"python\0-m\0nemotron_voicechat_runtime.cli\0status\0")
    assert not _owned_launcher_process(123, tmp_path)


def test_bot_source_snapshot_tracks_only_python_files(tmp_path: Path) -> None:
    package = tmp_path / "bot"
    package.mkdir()
    source = package / "demo.py"
    source.write_text("PROMPT = 'one'\n", encoding="utf-8")
    (package / "notes.txt").write_text("ignored\n", encoding="utf-8")
    first = _bot_source_snapshot(package)
    (package / "notes.txt").write_text("still ignored\n", encoding="utf-8")
    assert _bot_source_snapshot(package) == first
    source.write_text("PROMPT = 'two and longer'\n", encoding="utf-8")
    assert _bot_source_snapshot(package) != first


def test_restart_bot_signals_launcher_and_waits_for_new_listener(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    state_file = tmp_path / "runtime-state.json"
    state_file.write_text(
        json.dumps(
            {
                "schema": 2,
                "launcher_pid": 123,
                "launcher_start_time": "launcher-start",
                "pipecat_pid": 456,
                "pipecat_start_time": "old-bot-start",
                "bot_generation": 4,
                "pipecat_host": "127.0.0.1",
                "pipecat_port": 7860,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._owned_launcher_process", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._owned_pipecat_process", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._port_available", lambda _host, _port: False
    )

    def signal_launcher(pid: int, requested_signal: signal.Signals) -> None:
        assert pid == 123
        assert requested_signal == signal.SIGUSR1
        updated = json.loads(state_file.read_text(encoding="utf-8"))
        updated.update(
            {
                "pipecat_pid": 789,
                "pipecat_start_time": "new-bot-start",
                "bot_generation": 5,
            }
        )
        state_file.write_text(json.dumps(updated), encoding="utf-8")

    monkeypatch.setattr("nemotron_voicechat_runtime.cli.os.kill", signal_launcher)
    args = parser().parse_args(["--cache-root", str(tmp_path), "restart-bot", "--timeout", "1"])
    assert command_restart_bot(args) == 0
    assert "generation 5" in capsys.readouterr().out


def test_up_restarts_only_pipecat_when_restart_is_requested(tmp_path: Path, monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid
            self.returncode = None

        def poll(self):
            return None

    model = FakeProcess(100)
    bots = [FakeProcess(200), FakeProcess(201)]
    started_bots: list[FakeProcess] = []
    stopped_bots: list[FakeProcess] = []
    final_stacks = []
    signal_handlers = {
        signal.SIGTERM: signal.SIG_DFL,
        signal.SIGUSR1: signal.SIG_DFL,
    }
    sleeps = 0
    config = {
        "deployment": {
            "container_name": "voicechat-test",
            "model_port": 8786,
            "pipecat_host": "127.0.0.1",
            "pipecat_port": 7860,
            "health_timeout_seconds": 1,
        }
    }

    monkeypatch.setattr("nemotron_voicechat_runtime.cli.load_config", lambda _path: config)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli.default_layout",
        lambda *_args: Layout(tmp_path / "cache", tmp_path / "traces"),
    )
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._bootstrap_ready", lambda *_args: (True, "ready")
    )
    monkeypatch.setattr("nemotron_voicechat_runtime.cli._container_exists", lambda _name: False)
    monkeypatch.setattr("nemotron_voicechat_runtime.cli._port_available", lambda *_args: True)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli.model_container_command",
        lambda *_args, **_kwargs: ["model"],
    )
    monkeypatch.setattr("nemotron_voicechat_runtime.cli.subprocess.Popen", lambda _cmd: model)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._wait_health",
        lambda *_args: {"protocol": "strict-v3", "typed_input": {"ready": True}},
    )

    def start_bot(*_args):
        bot = bots[len(started_bots)]
        started_bots.append(bot)
        return bot

    monkeypatch.setattr("nemotron_voicechat_runtime.cli._start_pipecat", start_bot)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._stop_pipecat", lambda bot: stopped_bots.append(bot)
    )
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._stop_stack",
        lambda model_process, bot, name: final_stacks.append((model_process, bot, name)),
    )
    monkeypatch.setattr("nemotron_voicechat_runtime.cli._wait_ports_available", lambda *_args: None)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli.signal.getsignal", lambda sig: signal_handlers[sig]
    )

    def install_signal(sig, handler):
        signal_handlers[sig] = handler

    monkeypatch.setattr("nemotron_voicechat_runtime.cli.signal.signal", install_signal)

    def drive_loop(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            signal_handlers[signal.SIGUSR1](signal.SIGUSR1, None)
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr("nemotron_voicechat_runtime.cli.time.sleep", drive_loop)
    args = parser().parse_args(
        [
            "--cache-root",
            str(tmp_path / "cache"),
            "--trace-root",
            str(tmp_path / "traces"),
            "up",
        ]
    )

    assert command_up(args) == 0
    assert started_bots == bots
    assert stopped_bots == [bots[0]]
    assert final_stacks == [(model, bots[1], "voicechat-test")]


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

    monkeypatch.setattr("nemotron_voicechat_runtime.cli.socket.socket", lambda *_args: FakeSocket())

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


def test_docker_shutdown_uses_current_timeout_flag(monkeypatch) -> None:
    class ModelProcess:
        def poll(self):
            return None

        def wait(self, timeout):
            assert timeout == 25

    calls = []

    def run(command, check):
        calls.append((command, check))

    monkeypatch.setattr("nemotron_voicechat_runtime.cli.subprocess.run", run)
    _stop_stack(ModelProcess(), None, "voicechat-test")

    assert calls == [(["docker", "stop", "--timeout", "20", "voicechat-test"], False)]
