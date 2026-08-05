from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


def load_module():
    path = Path("tools/qualification/run_live_suite.py")
    spec = importlib.util.spec_from_file_location("run_live_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_up_command_is_foreground_and_loopback(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        cache_root=tmp_path / "cache",
        trace_root=tmp_path / "traces",
        model_port=9101,
        pipecat_port=9102,
    )
    command = module.up_command(args)
    assert command[-6:] == [
        "--model-port",
        "9101",
        "--host",
        "127.0.0.1",
        "--port",
        "9102",
    ]


def test_asr_command_is_offline_and_mounts_model_read_only(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        asr_model=tmp_path / "asr.nemo",
        runtime_image="runtime:test",
        output=tmp_path / "results",
    )
    command, _log = module.asr_command(args, tmp_path / "run", "asr.log")
    assert "--network" in command
    assert "none" in command
    assert f"{(tmp_path / 'asr.nemo').resolve()}:/models/asr.nemo:ro" in command


def test_component_contract_tests_run_offline_in_runtime_image(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    commands = []
    args = argparse.Namespace(output=tmp_path, runtime_image="runtime:test")

    def capture(command, **kwargs):
        commands.append((command, kwargs))

    monkeypatch.setattr(module, "run_checked", capture)
    module.run_component_contract_tests(args)
    command, kwargs = commands[0]
    assert command[:6] == ["docker", "run", "--rm", "--network", "none", "--workdir"]
    assert "--gpus" not in command
    assert command[-3:] == [
        "tests/conversion/test_nano_attribution.py",
        "tests/conversion/test_nano_replay.py",
        "tests/conversion/test_eartts_component_gate.py",
    ]
    assert kwargs["log"] == tmp_path / "component-contract-tests.log"


def test_gpu_component_prefix_uses_production_tokenizer_mount(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        cache_root=tmp_path / "cache",
        asr_model=tmp_path / "asr.nemo",
    )

    command = module.gpu_container_prefix(args)

    assert (
        f"{tmp_path / 'cache/artifacts/nano-skeleton'}:/models/NVIDIA-Nemotron-Nano-9B-v2:ro"
    ) in command
    assert not any("/models/nano-skeleton" in item for item in command)


def test_eartts_gate_receives_checkpoint_pinned_tokenizer_path(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    commands = []
    output = tmp_path / "run"
    output.mkdir()
    args = argparse.Namespace(
        output=output,
        cache_root=tmp_path / "cache",
        asr_model=tmp_path / "asr.nemo",
        runtime_image="runtime:test",
    )

    monkeypatch.setattr(module, "extract_replay_corpus", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "run_checked",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    module.run_component_gates(args)

    eager_command = commands[1][0]
    index = eager_command.index("--nano-skeleton")
    assert eager_command[index + 1] == "/models/NVIDIA-Nemotron-Nano-9B-v2"


def test_generated_fixture_manifest_is_remapped_from_container_paths(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    output = tmp_path / "run"
    output.mkdir()
    args = argparse.Namespace(
        fixture_manifest=None,
        browser_mic_wav=None,
        output=output,
        cache_root=tmp_path / "cache",
        runtime_image="runtime:test",
    )

    def generate(_command, **_kwargs):
        fixtures = output / "fixtures"
        fixtures.mkdir()
        payload = {
            "turns": [{"text": "hello", "audio_path": "/qualification/fixtures/turn-01.wav"}],
            "browser_microphone_wav": "/qualification/fixtures/browser-microphone.wav",
        }
        (fixtures / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module, "run_checked", generate)
    module.prepare_fixtures(args)
    assert args.fixture_manifest == output / "fixture-manifest-host.json"
    manifest = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    assert manifest["turns"][0]["audio_path"] == str(output / "fixtures/turn-01.wav")
    assert args.browser_mic_wav == output / "fixtures/browser-microphone.wav"


def test_browser_gate_records_failure_then_success(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = argparse.Namespace(
        python=Path("/runtime/python"), output=tmp_path, model_port=8786
    )
    returncodes = iter((1, 0))
    idle_checks = []

    def run(command, **kwargs):
        Path(kwargs["stdout"].name).write_text("browser evidence\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, next(returncodes))

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "wait_model_idle", lambda port: idle_checks.append(port))

    report = module.run_browser_gate(args, {"TEST": "1"})

    assert report["passed"] is True
    assert [attempt["passed"] for attempt in report["attempts"]] == [False, True]
    assert idle_checks == [8786]
    retained = json.loads((tmp_path / "browser-attempts.json").read_text())
    assert retained == report
    assert all(attempt["log_sha256"] for attempt in retained["attempts"])


def test_browser_gate_fails_after_two_recorded_attempts(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = argparse.Namespace(
        python=Path("/runtime/python"), output=tmp_path, model_port=8786
    )

    def run(command, **kwargs):
        Path(kwargs["stdout"].name).write_text("failed browser evidence\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "wait_model_idle", lambda _port: None)

    try:
        module.run_browser_gate(args, {})
    except RuntimeError as exc:
        assert "failed 2 recorded attempts" in str(exc)
    else:
        raise AssertionError("two failed browser attempts must fail the gate")

    report = json.loads((tmp_path / "browser-attempts.json").read_text())
    assert report["passed"] is False
    assert len(report["attempts"]) == 2
