from __future__ import annotations

import hashlib
import json
import signal
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron_voicechat_runtime.artifacts import PC2A_ENVIRONMENT, Layout, load_config
from nemotron_voicechat_runtime.cli import (
    _activate_converted_release,
    _asr_provenance_contract,
    _bootstrap_ready,
    _bot_source_snapshot,
    _build_image,
    _ensure_asr_evaluator,
    _finalize_source_asr_stage,
    _owned_launcher_process,
    _owned_pipecat_process,
    _process_start_time,
    _remove_stale_conversion_activations,
    _runtime_image_matches_checkout,
    _stop_stack,
    _tree_copy_gib,
    _validated_external_asr_report,
    _wait_ports_available,
    command_bootstrap,
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


def test_asr_evaluator_audits_and_returns_one_immutable_resolution(monkeypatch) -> None:
    image_id = f"sha256:{'a' * 64}"
    audited: list[list[str]] = []
    inspections = 0

    def run(command, **_kwargs):
        nonlocal inspections
        if command[:3] == ["docker", "image", "inspect"]:
            inspections += 1
            return SimpleNamespace(stdout=image_id + "\n")
        raise AssertionError(command)

    monkeypatch.setattr("nemotron_voicechat_runtime.cli._image_exists", lambda _image: True)
    monkeypatch.setattr("nemotron_voicechat_runtime.cli.subprocess.run", run)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._run",
        lambda command, **_kwargs: audited.append(command),
    )

    assert _ensure_asr_evaluator(load_config(), offline=False, no_cache=False) == image_id
    assert inspections == 1
    assert audited[0][-1] == image_id


def test_runtime_image_source_identity_matches_exact_checkout(monkeypatch) -> None:
    digest = "a" * 64
    for function in (
        "public_runtime_source_sha256",
        "public_runtime_payload_sha256",
        "public_runtime_recipe_sha256",
    ):
        monkeypatch.setattr(f"nemotron_voicechat_runtime.cli.{function}", lambda _root: digest)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(
            stdout=json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "ai.pipecat.voicechat.project-source-sha256": digest,
                                "ai.pipecat.voicechat.project-payload-sha256": digest,
                                "ai.pipecat.voicechat.build-recipe-sha256": digest,
                            }
                        }
                    }
                ]
            )
        ),
    )

    assert _runtime_image_matches_checkout("runtime:test") is True


def test_bootstrap_rebuilds_stale_runtime_and_offline_fails(tmp_path: Path, monkeypatch) -> None:
    config = {"image": {"runtime": "runtime:test"}}
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    commands: list[list[str]] = []
    monkeypatch.setattr("nemotron_voicechat_runtime.cli._image_exists", lambda _image: True)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._runtime_image_matches_checkout",
        lambda _image: False,
    )
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._run",
        lambda command, **_kwargs: commands.append(command),
    )

    _build_image(config, layout, offline=False, no_cache=False)

    assert commands == [[str(Path.cwd() / "container/build-public-runtime.sh")]]
    with pytest.raises(RuntimeError, match="stale image"):
        _build_image(config, layout, offline=True, no_cache=False)


def test_up_readiness_rejects_runtime_from_older_checkout(tmp_path: Path, monkeypatch) -> None:
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    layout.cache.mkdir(parents=True)
    release = {"release_sha256": "a" * 64, "revision": "revision"}
    (layout.cache / "artifact-state.json").write_text(
        json.dumps(
            {
                "release_sha256": release["release_sha256"],
                "release_revision": release["revision"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("nemotron_voicechat_runtime.cli._image_exists", lambda _image: True)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._runtime_image_matches_checkout",
        lambda _image: False,
    )

    ready, reason = _bootstrap_ready(
        {"image": {"runtime": "runtime:test"}, "artifacts": {"release": release}},
        layout,
    )

    assert ready is False
    assert reason == "public runtime image does not match this checkout"


def test_bootstrap_propagates_audited_asr_id_to_source_conversion(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    layout.cache.mkdir(parents=True)
    asr_id = f"sha256:{'a' * 64}"
    runtime_id = f"sha256:{'b' * 64}"
    observed: list[str] = []
    for name in (
        "require_free_space",
        "_executable",
        "_sync_host",
        "_download_artifacts",
        "_build_image",
    ):
        monkeypatch.setattr(f"nemotron_voicechat_runtime.cli.{name}", lambda *_a, **_k: None)
    monkeypatch.setattr("nemotron_voicechat_runtime.cli.load_config", lambda _path: config)
    monkeypatch.setattr("nemotron_voicechat_runtime.cli.default_layout", lambda *_a: layout)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli._ensure_asr_evaluator", lambda *_a, **_k: asr_id
    )

    def convert(_args, _config, _layout, exact_asr_id):
        observed.append(exact_asr_id)
        return {"artifact_materialization": "source-converted"}

    monkeypatch.setattr("nemotron_voicechat_runtime.cli._convert_from_source", convert)
    monkeypatch.setattr(
        "nemotron_voicechat_runtime.cli.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout=runtime_id + "\n"),
    )
    args = SimpleNamespace(
        config=Path("config/production-candidate-1.toml"),
        cache_root=layout.cache,
        trace_root=layout.traces,
        offline=False,
        no_cache=False,
        convert_from_source=True,
    )

    assert command_bootstrap(args) == 0
    assert observed == [asr_id]


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
    assert f"VOICECHAT_RUNTIME_IMAGE={config['image']['runtime']}" in command
    assert "VOICECHAT_RUNTIME_IMAGE_ID" not in rendered
    assert f"{layout.hf_home}:/models/huggingface:ro" in command
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in rendered
    assert "API_KEY" not in rendered


def test_model_command_records_immutable_runtime_image_id(tmp_path: Path) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")

    command = model_container_command(
        config,
        layout,
        9876,
        runtime_image_id=f"sha256:{'a' * 64}",
    )

    assert f"VOICECHAT_RUNTIME_IMAGE_ID=sha256:{'a' * 64}" in command


def test_model_command_enables_pad_pair_trace_only_when_requested(tmp_path: Path) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")

    ordinary = " ".join(model_container_command(config, layout, 9876))
    traced = model_container_command(config, layout, 9876, trace_pad_pair=True)

    assert "VOICECHAT_NANO_PAD_PAIR_TRACE" not in ordinary
    assert "VOICECHAT_NANO_PAD_PAIR_TRACE=1" in traced


def test_model_command_enables_bounded_post_fc_trace_only_when_requested(
    tmp_path: Path,
) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")

    ordinary = " ".join(model_container_command(config, layout, 9876))
    traced = model_container_command(config, layout, 9876, trace_post_fc_tokens=9)

    assert "S2S_POST_FC_TOKEN_TRACE_FRAMES" not in ordinary
    assert "S2S_POST_FC_TOKEN_TRACE_FRAMES=9" in traced
    with pytest.raises(ValueError, match="0 through 32"):
        model_container_command(config, layout, 9876, trace_post_fc_tokens=33)


def test_model_command_enables_agent_logits_only_with_post_fc_trace(
    tmp_path: Path,
) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")

    ordinary = " ".join(model_container_command(config, layout, 9876))
    traced = " ".join(
        model_container_command(
            config,
            layout,
            9876,
            trace_post_fc_tokens=9,
            trace_post_fc_agent_logits=20,
        )
    )

    assert "S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK" not in ordinary
    assert "S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK=20" in traced
    with pytest.raises(ValueError, match="requires a nonzero"):
        model_container_command(config, layout, 9876, trace_post_fc_agent_logits=20)
    with pytest.raises(ValueError, match="0 through 20"):
        model_container_command(
            config,
            layout,
            9876,
            trace_post_fc_tokens=9,
            trace_post_fc_agent_logits=21,
        )


def test_model_command_enables_fc_async_heartbeat_only_when_requested(
    tmp_path: Path,
) -> None:
    config = load_config()
    layout = Layout(tmp_path / "cache", tmp_path / "traces")

    ordinary = " ".join(model_container_command(config, layout, 9876))
    traced = " ".join(model_container_command(config, layout, 9876, trace_fc_async_heartbeat=True))

    assert "S2S_FC_ASYNC_HEARTBEAT" not in ordinary
    assert "S2S_FC_ASYNC_HEARTBEAT=1" in traced


def test_cli_exposes_stable_foreground_commands() -> None:
    cli = parser()
    assert cli.parse_args(["bootstrap", "--offline"]).offline is True
    conversion = cli.parse_args(["bootstrap", "--convert-from-source"])
    assert conversion.convert_from_source is True
    up = cli.parse_args(
        [
            "up",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload-bot",
            "--trace-pad-pair",
            "--trace-post-fc-tokens",
            "9",
            "--trace-post-fc-agent-logits",
            "20",
            "--trace-fc-async-heartbeat",
        ]
    )
    assert up.port == 9000
    assert up.reload_bot is True
    assert up.trace_post_fc_agent_logits == 20
    assert up.trace_pad_pair is True
    assert up.trace_post_fc_tokens == 9
    assert up.trace_fc_async_heartbeat is True
    assert cli.parse_args(["restart-bot"]).timeout == 45.0
    assert cli.parse_args(["test", "--live"]).live is True
    assert cli.parse_args(["down"]).command == "down"
    assert cli.parse_args(["status"]).command == "status"
    assert cli.parse_args(["release", "stage"]).arguments == ["stage"]


def test_source_conversion_atomically_activates_verified_local_files(
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
    evaluator_id = f"sha256:{'a' * 64}"
    asr_evidence = {}
    for name in ("eartts-eager", "eartts-graph"):
        source = converted.parent / f"gates/{name}"
        source.mkdir(parents=True)
        input_records = {}
        results = {}
        for channel in ("streaming", "offline"):
            wav = source / f"{channel}.wav"
            with wave.open(str(wav), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\0\0" * 160)
            observed = {
                "path": wav.name,
                "bytes": wav.stat().st_size,
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            }
            input_records[channel] = observed
            results[channel] = {
                "input_bytes": observed["bytes"],
                "input_sha256": observed["sha256"],
            }
        source_report = {
            "asr_inputs": input_records,
            "fixture": {"reference_text": "The answer is five."},
            "external_asr": {"status": "pending", "max_wer": 0.5},
        }
        source_report_path = source / "report.json"
        source_report_path.write_text(json.dumps(source_report), encoding="utf-8")
        final = {
            "passed": True,
            "pending_external_asr": False,
            "pre_asr_report_sha256": hashlib.sha256(source_report_path.read_bytes()).hexdigest(),
            "external_asr": {
                **_asr_provenance_contract(),
                **results["streaming"],
                "passed": True,
                "evaluator_image_id": evaluator_id,
                "transcript": "The answer is five.",
                "wer": 0.0,
                "max_wer": 0.5,
            },
            "offline_diagnostic": {"external_asr": results["offline"]},
        }
        final_path = converted.parent / f"external-asr/{name}.json"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(json.dumps(final), encoding="utf-8")
        asr_evidence[name] = _validated_external_asr_report(
            converted.parent, final_path, source, evaluator_id
        )

    eager_source = converted.parent / "gates/eartts-eager"
    eager_final_path = converted.parent / "external-asr/eartts-eager.json"
    original_final = json.loads(eager_final_path.read_text(encoding="utf-8"))
    for field, value, message in (
        ("backend_sha256", "f" * 64, "audited evaluator"),
        ("transcript", "", "does not rederive"),
        ("max_wer", 0.9, "does not rederive"),
        ("passed", False, "invalid finalized"),
    ):
        mutated = json.loads(json.dumps(original_final))
        target = mutated if field == "passed" else mutated["external_asr"]
        target[field] = value
        eager_final_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            _validated_external_asr_report(
                converted.parent, eager_final_path, eager_source, evaluator_id
            )
    eager_final_path.write_text(json.dumps(original_final), encoding="utf-8")

    source_report_path = eager_source / "report.json"
    original_source = json.loads(source_report_path.read_text(encoding="utf-8"))
    streaming_wav = eager_source / "streaming.wav"
    original_wav = streaming_wav.read_bytes()
    with wave.open(str(streaming_wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 80)
    changed = {
        "path": "streaming.wav",
        "bytes": streaming_wav.stat().st_size,
        "sha256": hashlib.sha256(streaming_wav.read_bytes()).hexdigest(),
    }
    mutated_source = json.loads(json.dumps(original_source))
    mutated_source["asr_inputs"]["streaming"] = changed
    source_report_path.write_text(json.dumps(mutated_source), encoding="utf-8")
    mutated_final = json.loads(json.dumps(original_final))
    mutated_final["pre_asr_report_sha256"] = hashlib.sha256(
        source_report_path.read_bytes()
    ).hexdigest()
    mutated_final["external_asr"]["input_bytes"] = changed["bytes"]
    mutated_final["external_asr"]["input_sha256"] = changed["sha256"]
    eager_final_path.write_text(json.dumps(mutated_final), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not PCM16 mono 16 kHz"):
        _validated_external_asr_report(
            converted.parent, eager_final_path, eager_source, evaluator_id
        )
    streaming_wav.write_bytes(original_wav)
    source_report_path.write_text(json.dumps(original_source), encoding="utf-8")
    eager_final_path.write_text(json.dumps(original_final), encoding="utf-8")

    assert _finalize_source_asr_stage(converted.parent, evaluator_id) == asr_evidence
    assert _finalize_source_asr_stage(converted.parent, evaluator_id) == asr_evidence
    reproduction = {
        "schema": 1,
        "kind": "voicechat_release_artifact_reproduction",
        "asr_evaluator_image_id": evaluator_id,
        "component_gates_external_asr": asr_evidence,
    }
    reproduction_sha = hashlib.sha256(
        json.dumps(reproduction, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reproduction["sha256"] = reproduction_sha
    attestation = converted.parent / "external-asr/reproduction.json"
    attestation.write_text(json.dumps(reproduction), encoding="utf-8")

    verified: list[Path] = []

    def verify(root: Path, expected_sha: str):
        assert expected_sha == "signed-release-sha"
        assert (root / "nano/model.safetensors").read_bytes() == b"source-nano"
        assert (root / "eartts/model.safetensors").read_bytes() == b"source-eartts"
        assert (root / "manifests/release.json").read_text() == "signed\n"
        verified.append(root)
        return {"release_sha256": expected_sha}

    monkeypatch.setattr("nemotron_voicechat_runtime.bootstrap_download.verify_release", verify)
    result = _activate_converted_release(layout, converted, "signed-release-sha", evaluator_id)

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
    attestation = converted.parent / "external-asr/reproduction.json"
    attestation.parent.mkdir(parents=True)
    attestation.write_text(
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
        _activate_converted_release(layout, converted, "signed-release-sha", f"sha256:{'a' * 64}")


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
    layout = Layout(tmp_path / "cache", tmp_path / "traces")
    layout.state_file.parent.mkdir(parents=True)
    layout.state_file.write_text(
        json.dumps({"runtime_image_id": f"sha256:{'a' * 64}"}), encoding="utf-8"
    )
    monkeypatch.setattr("nemotron_voicechat_runtime.cli.default_layout", lambda *_args: layout)
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
