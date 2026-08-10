from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def load_module():
    path = Path("tools/qualification/run_live_suite.py")
    spec = importlib.util.spec_from_file_location("run_live_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_live_suite_fails_fast_when_playwright_chromium_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()

    class FakePlaywright:
        chromium = type("Chromium", (), {"executable_path": str(tmp_path / "missing")})()

    class FakeContext:
        def __enter__(self):
            return FakePlaywright()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: FakeContext())
    with pytest.raises(RuntimeError, match="playwright install chromium"):
        module.require_browser_executable()


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
    traced = module.up_command(args, trace_fc_async_heartbeat=True)
    assert traced[-1] == "--trace-fc-async-heartbeat"
    assert "--trace-fc-async-heartbeat" not in command


def test_asr_command_is_offline_and_uses_immutable_evaluator(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        asr_evaluator_image_id=f"sha256:{'a' * 64}",
        output=tmp_path / "results",
    )
    command, _log = module.asr_command(args, tmp_path / "run", "asr.log")
    assert "--network" in command
    assert "none" in command
    assert "PYTHONPATH=/workspace/project/src" not in command
    assert command[command.index("python3") + 1] == "-P"
    assert f"VOICECHAT_ASR_EVALUATOR_IMAGE_ID=sha256:{'a' * 64}" in command
    assert f"sha256:{'a' * 64}" in command
    assert not any("asr.nemo" in item for item in command)


def test_memory_matrix_asr_disallows_silent_responses(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        asr_evaluator_image_id=f"sha256:{'a' * 64}",
        output=tmp_path / "results",
    )

    command, _log = module.asr_command(
        args,
        tmp_path / "memory-matrix",
        "memory-matrix-asr.log",
        max_silent_rate=0.0,
    )

    assert command[-2:] == ["--max-silent-rate", "0.0"]


def test_parity_asr_preserves_raw_arm_and_writes_separate_evaluation(
    tmp_path: Path,
) -> None:
    module = load_module()
    args = argparse.Namespace(
        asr_evaluator_image_id=f"sha256:{'a' * 64}",
        output=tmp_path / "results",
    )
    raw = tmp_path / "runtime/typed"
    evaluated = tmp_path / "asr/typed"

    command, _log = module.asr_command(
        args,
        raw,
        "typed-asr.log",
        max_silent_rate=0.0,
        output_dir=evaluated,
    )

    assert f"{raw.resolve()}:/qualification-input:ro" in command
    assert f"{evaluated.resolve()}:/qualification-output" in command
    assert command[command.index("--run-dir") + 1] == "/qualification-input"
    assert command[command.index("--output-dir") + 1] == "/qualification-output"
    assert command[-2:] == ["--max-silent-rate", "0.0"]


def test_diagnostic_runner_keys_exit_to_completeness_not_passed(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    report_path = tmp_path / "diagnostic.json"
    payload = {
        "passed": False,
        "promotion_eligible": False,
        "diagnostic_complete": True,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )

    report, returncode = module.run_diagnostic_report_command(
        ["diagnostic"],
        log=tmp_path / "diagnostic.log",
        report_path=report_path,
        timeout=1,
    )

    assert returncode == 0
    assert report["diagnostic_complete"] is True
    assert report["passed"] is False

    payload["passed"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract"):
        module.run_diagnostic_report_command(
            ["diagnostic"],
            log=tmp_path / "diagnostic-red.log",
            report_path=report_path,
            timeout=1,
        )


def test_tool_freshness_commands_pin_images_identity_and_order(tmp_path: Path) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    args = argparse.Namespace(
        python=tmp_path / "python",
        output=tmp_path / "run",
        cache_root=tmp_path / "cache",
        runtime_image="runtime:qualified",
        model_port=9123,
        reverse_forward_comparison=tmp_path / "forward/comparison.json",
        reverse_forward_fixture=tmp_path / "forward/fixtures/manifest.json",
    )

    fixture = module.tool_freshness_fixture_command(args, runtime_image_id=image_id)
    arm = module.tool_freshness_arm_command(args, runtime_image_id=image_id)
    compare = module.tool_freshness_compare_command(args)
    reverse_arm = module.tool_freshness_arm_command(
        args, runtime_image_id=image_id, nonpromotable_reverse_order=True
    )
    reverse_preflight = module.tool_freshness_reverse_analyzer_command(
        args, preflight_only=True
    )
    reverse_analyze = module.tool_freshness_reverse_analyzer_command(args)

    assert fixture[fixture.index("--network") + 1] == "none"
    assert "PYTHONPATH=/workspace/project/src" in fixture
    assert "HF_HOME=/models/huggingface" in fixture
    assert "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub" in fixture
    assert (
        f"{(tmp_path / 'cache/huggingface').resolve()}:/models/huggingface:ro"
        in fixture
    )
    assert image_id in fixture
    assert fixture[fixture.index("--runtime-image") + 1] == "runtime:qualified"
    assert fixture[fixture.index("--runtime-image-id") + 1] == image_id
    assert arm[arm.index("--runtime-image-id") + 1] == image_id
    assert arm[arm.index("--url") + 1] == "ws://127.0.0.1:9123/v1/realtime"
    assert "--nonpromotable-reverse-order" not in arm
    assert reverse_arm[-1] == "--nonpromotable-reverse-order"
    assert compare[compare.index("--runtime-report") + 1] == str(
        args.output / "runtime/report.json"
    )
    assert compare[compare.index("--policy") + 1] == "parity_margin_v2"
    assert "--preflight-only" in reverse_preflight
    assert "--reverse-runtime-report" not in reverse_preflight
    assert reverse_analyze[reverse_analyze.index("--reverse-runtime-report") + 1] == str(
        args.output / "runtime/report.json"
    )


@pytest.mark.parametrize("comparison_passed", [True, False])
def test_tool_freshness_live_orchestration_orders_gpu_owners_and_gates(
    tmp_path: Path, monkeypatch, comparison_passed: bool
) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    order: list[str] = []

    class FakeStack:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            order.append("stack-stop")
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    args = argparse.Namespace(
        python=tmp_path / "python",
        output=tmp_path / "run",
        runtime_image="runtime:qualified",
        asr_evaluator_image_id=module.PROMOTED_ASR_EVALUATOR_IMAGE_ID,
        cache_root=tmp_path / "cache",
        trace_root=tmp_path / "trace",
        model_port=8786,
        pipecat_port=7860,
        startup_timeout=60,
        ready_budget_seconds=120,
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: {
            "image": {"runtime": args.runtime_image},
            "deployment": {"container_name": "voicechat-test-model"},
        },
    )
    monkeypatch.setattr(module, "runtime_image_id", lambda _image: image_id)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: FakeStack())
    monkeypatch.setattr(module, "wait_for_playground", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "wait_model_idle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "validate_tool_freshness_fixtures",
        lambda *_args, **_kwargs: ({"sha256": "c" * 64}, (b"pcm",) * 4),
    )
    monkeypatch.setattr(module, "validate_fixture_source_provenance", lambda *_a, **_k: None)

    def checked(command, **_kwargs):
        if any(item.endswith("generate_tool_freshness_fixtures.py") for item in command):
            order.append("fixture")
        elif any(item.endswith("tool_freshness_parity.py") for item in command):
            order.append("runtime-arms")
            report = args.output / "runtime/report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"passed": True}), encoding="utf-8")
        else:
            raise AssertionError(command)

    def json_gate(command, *, report_path, **_kwargs):
        if any(item.endswith("transcribe_sustained.py") for item in command):
            modality = report_path.parent.name
            order.append(f"asr-{modality}")
            payload = {"passed": True, "modality": modality}
        else:
            order.append("compare")
            payload = {
                "passed": comparison_passed,
                "kind": "tool_freshness_input_modality_parity_v2",
                "core_passed": True,
                "attended_review_required": not comparison_passed,
                "unattended_promotion_eligible": comparison_passed,
            }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload, 0 if comparison_passed else 1

    monkeypatch.setattr(module, "run_checked", checked)
    monkeypatch.setattr(module, "run_json_report_command", json_gate)

    report = module.run_tool_freshness_parity_only(args)

    assert report["passed"] is comparison_passed
    assert report["comparison"]["kind"] == "tool_freshness_input_modality_parity_v2"
    assert order == [
        "fixture",
        "runtime-arms",
        "stack-stop",
        "asr-typed",
        "asr-speech",
        "compare",
    ]
    orchestration = json.loads((args.output / "orchestration.json").read_text())
    assert [stage["name"] for stage in orchestration["stages"]] == [
        "fixture",
        "runtime_arms",
        "asr_typed",
        "asr_speech",
        "compare",
    ]
    assert all(stage["status"] == "completed" for stage in orchestration["stages"])
    assert orchestration["stages"][-1]["returncode"] == (
        0 if comparison_passed else 1
    )


def test_reverse_diagnostic_is_nonpromotable_and_runs_speech_then_typed(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    order: list[str] = []

    class FakeStack:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            order.append("stack-stop")
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    args = argparse.Namespace(
        python=tmp_path / "python",
        output=tmp_path / "run",
        runtime_image="runtime:qualified",
        asr_evaluator_image_id=module.PROMOTED_ASR_EVALUATOR_IMAGE_ID,
        cache_root=tmp_path / "cache",
        trace_root=tmp_path / "trace",
        model_port=8786,
        pipecat_port=7860,
        startup_timeout=60,
        ready_budget_seconds=120,
        reverse_forward_comparison=tmp_path / "forward/comparison.json",
        reverse_forward_fixture=tmp_path / "forward/fixtures/manifest.json",
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: {
            "image": {"runtime": args.runtime_image},
            "deployment": {"container_name": "voicechat-test-model"},
        },
    )
    monkeypatch.setattr(module, "runtime_image_id", lambda _image: image_id)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_a, **_k: FakeStack())
    monkeypatch.setattr(module, "wait_for_playground", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "wait_model_idle", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "validate_tool_freshness_fixtures",
        lambda *_a, **_k: ({"sha256": "c" * 64}, (b"pcm",) * 4),
    )
    monkeypatch.setattr(module, "validate_fixture_source_provenance", lambda *_a, **_k: None)

    def checked(command, **_kwargs):
        if any(item.endswith("generate_tool_freshness_fixtures.py") for item in command):
            order.append("fixture")
        elif any(item.endswith("tool_freshness_parity.py") for item in command):
            assert command[-1] == "--nonpromotable-reverse-order"
            order.append("runtime-speech-typed")
            report_path = args.output / "runtime/report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "promotion_eligible": False,
                        "execution_order": ["speech", "typed"],
                    }
                ),
                encoding="utf-8",
            )
        else:
            raise AssertionError(command)

    def json_gate(command, *, report_path, **_kwargs):
        if "--preflight-only" in command:
            order.append("preflight")
            payload = {
                "passed": False,
                "promotion_eligible": False,
                "diagnostic_complete": True,
            }
        elif any(item.endswith("transcribe_sustained.py") for item in command):
            modality = report_path.parent.name
            order.append(f"asr-{modality}")
            payload = {"passed": True, "modality": modality}
        else:
            order.append("analyze")
            payload = {
                "passed": False,
                "promotion_eligible": False,
                "diagnostic_complete": True,
            }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload, 0

    monkeypatch.setattr(module, "run_checked", checked)
    monkeypatch.setattr(module, "run_json_report_command", json_gate)
    monkeypatch.setattr(module, "run_diagnostic_report_command", json_gate)

    report = module.run_tool_freshness_reverse_diagnostic_only(args)

    assert report["passed"] is False
    assert report["promotion_eligible"] is False
    assert report["diagnostic_complete"] is True
    assert order == [
        "preflight",
        "fixture",
        "runtime-speech-typed",
        "stack-stop",
        "asr-speech",
        "asr-typed",
        "analyze",
    ]


def test_thread_dump_request_signals_container_and_verifies_python_dump(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    log_path = tmp_path / "stack.log"
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        with log_path.open("ab") as output:
            output.write(b"Current thread 0x0000 (most recent call first):\n")
        return subprocess.CompletedProcess(command, 0, "voicechat-test-model\n", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    with log_path.open("wb") as stack_log:
        result = module.request_model_thread_dump(
            container_name="voicechat-test-model",
            stack_log_path=log_path,
            stack_log=stack_log,
            wait_seconds=0.1,
        )

    assert calls == [["docker", "kill", "--signal=USR1", "voicechat-test-model"]]
    assert result["returncode"] == 0
    assert result["dump_observed"] is True
    with pytest.raises(ValueError, match="not safe"):
        with log_path.open("ab") as stack_log:
            module.request_model_thread_dump(
                container_name="../wrong",
                stack_log_path=log_path,
                stack_log=stack_log,
            )


@pytest.mark.parametrize(
    "arm_failure",
    [
        RuntimeError("arm failed"),
        subprocess.TimeoutExpired(["tool-freshness-arm"], 1200),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_tool_freshness_failure_dumps_threads_before_stack_teardown(
    tmp_path: Path, monkeypatch, arm_failure: Exception, reverse: bool
) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    order: list[str] = []
    started_commands: list[list[str]] = []

    class FakeStack:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            order.append("stack-stop")
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    args = argparse.Namespace(
        python=tmp_path / "python",
        output=tmp_path / "run",
        runtime_image="runtime:qualified",
        asr_evaluator_image_id=module.PROMOTED_ASR_EVALUATOR_IMAGE_ID,
        cache_root=tmp_path / "cache",
        trace_root=tmp_path / "trace",
        model_port=8786,
        pipecat_port=7860,
        startup_timeout=60,
        ready_budget_seconds=120,
        reverse_forward_comparison=tmp_path / "forward/comparison.json",
        reverse_forward_fixture=tmp_path / "forward/fixtures/manifest.json",
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: {
            "image": {"runtime": args.runtime_image},
            "deployment": {"container_name": "voicechat-test-model"},
        },
    )
    monkeypatch.setattr(module, "runtime_image_id", lambda _image: image_id)

    def popen(command, **_kwargs):
        started_commands.append(command)
        return FakeStack()

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "wait_for_playground", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "wait_model_idle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "validate_tool_freshness_fixtures",
        lambda *_args, **_kwargs: ({"sha256": "c" * 64}, (b"pcm",) * 4),
    )
    monkeypatch.setattr(module, "validate_fixture_source_provenance", lambda *_a, **_k: None)

    def checked(command, **_kwargs):
        if any(item.endswith("generate_tool_freshness_fixtures.py") for item in command):
            order.append("fixture")
            return
        order.append("runtime-arms")
        raise arm_failure

    monkeypatch.setattr(module, "run_checked", checked)

    def diagnostic(_command, *, report_path, **_kwargs):
        order.append("preflight")
        payload = {
            "passed": False,
            "promotion_eligible": False,
            "diagnostic_complete": True,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload, 0

    monkeypatch.setattr(module, "run_diagnostic_report_command", diagnostic)

    def dump(**kwargs):
        order.append("thread-dump")
        assert kwargs["container_name"] == "voicechat-test-model"
        return {
            "command": ["docker", "kill", "--signal=USR1", "voicechat-test-model"],
            "returncode": 0,
            "dump_observed": True,
        }

    monkeypatch.setattr(module, "request_model_thread_dump", dump)

    with pytest.raises(type(arm_failure)):
        if reverse:
            module.run_tool_freshness_reverse_diagnostic_only(args)
        else:
            module.run_tool_freshness_parity_only(args)

    assert order == (
        ["preflight", "fixture", "runtime-arms", "thread-dump", "stack-stop"]
        if reverse
        else ["fixture", "runtime-arms", "thread-dump", "stack-stop"]
    )
    assert started_commands[0][-1] == "--trace-fc-async-heartbeat"
    orchestration = json.loads((args.output / "orchestration.json").read_text())
    runtime_stage = orchestration["stages"][-1]
    assert runtime_stage["status"] == "failed"
    assert runtime_stage["thread_dump"]["dump_observed"] is True


def test_direct_semantic_probe_commands_pin_corpus_image_and_precision(tmp_path: Path) -> None:
    module = load_module()
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}", encoding="utf-8")
    fp32 = tmp_path / "fp32"
    args = argparse.Namespace(
        cache_root=tmp_path / "cache",
        output=tmp_path / "output",
        runtime_image="runtime:test",
        fp32_derived_root=fp32,
        direct_semantic_corpus=corpus,
    )
    image_id = f"sha256:{'a' * 64}"

    fp32_command = module.direct_semantic_probe_command(
        args, precision="fp32", runtime_image_id=image_id
    )
    w8_command = module.direct_semantic_probe_command(
        args, precision="w8", runtime_image_id=image_id
    )
    speech_root = tmp_path / "Speech"
    trace_command = module.direct_semantic_probe_command(
        argparse.Namespace(**vars(args), diagnostic_speech_root=speech_root),
        precision="w8",
        runtime_image_id=image_id,
        output_relative=Path("trace-on"),
        token_forms=("text",),
        function_logit_trace_top_k=10,
    )

    assert f"{fp32.resolve()}:/derived:ro" in fp32_command
    relocated = (args.output / "direct-text-semantics/fp32-relocated-manifest.json").resolve()
    assert f"{relocated}:/qualification-input/fp32-manifest.json:ro" in fp32_command
    assert "/qualification-input/fp32-manifest.json" in fp32_command
    assert "/derived/manifest.json" not in fp32_command
    assert "VOICECHAT_DIRECT_SEMANTIC_PRECISION=fp32" in fp32_command
    assert f"VOICECHAT_RUNTIME_IMAGE_ID={image_id}" in fp32_command
    assert "VOICECHAT_NANO_PAD_PAIR=0" in fp32_command
    assert f"{corpus.resolve()}:/qualification-input/corpus.json:ro" in fp32_command
    assert "/models/derived/manifests/release.json" in w8_command
    assert "VOICECHAT_DIRECT_SEMANTIC_PRECISION=w8" in w8_command
    assert "GIT_CONFIG_COUNT=1" not in fp32_command
    assert "GIT_CONFIG_COUNT=1" not in w8_command
    assert "PYTHONPATH=/workspace/project/src:/opt/Speech" not in fp32_command
    assert "PYTHONPATH=/workspace/project/src:/opt/Speech" not in w8_command
    assert "--network" in fp32_command and "none" in fp32_command
    assert "S2S_FUNCTION_LOGIT_TRACE_TOPK=10" in trace_command
    assert "VOICECHAT_DIRECT_SEMANTIC_FORMS=text" in trace_command
    assert f"{speech_root.resolve()}:/opt/Speech:ro" in trace_command
    assert "GIT_CONFIG_COUNT=1" in trace_command
    assert "GIT_CONFIG_KEY_0=safe.directory" in trace_command
    assert "GIT_CONFIG_VALUE_0=/opt/Speech" in trace_command
    assert "PYTHONPATH=/workspace/project/src:/opt/Speech" in trace_command
    assert f"{(args.output / 'trace-on').resolve()}:/qualification" in trace_command


def test_fp32_qualification_manifest_relocates_only_component_paths(
    tmp_path: Path,
) -> None:
    module = load_module()
    source = tmp_path / "manifest.json"
    destination = tmp_path / "output/relocated.json"
    manifest = {
        "schema": 1,
        "kind": "exact_public_vllm_extraction",
        "source": {"repository": "repo", "revision": "rev"},
        "components": {
            name: {
                "path": paths["source"],
                "model": {"sha256": name * 8, "bytes": len(name)},
                "files": {"config.json": {"sha256": "abc", "bytes": 3}},
                "reproducibility": {"byte_identical": True, "runs": 2},
            }
            for name, paths in module.FP32_QUALIFICATION_COMPONENT_PATHS.items()
        },
    }
    source.write_text(json.dumps(manifest), encoding="utf-8")
    source_before = source.read_bytes()

    relocated = module.write_fp32_qualification_manifest(source, destination)

    assert source.read_bytes() == source_before
    restored = copy.deepcopy(relocated)
    audit = restored.pop("qualification_manifest_relocation")
    for name, paths in module.FP32_QUALIFICATION_COMPONENT_PATHS.items():
        assert restored["components"][name]["path"] == paths["container"]
        restored["components"][name]["path"] = paths["source"]
    assert restored == manifest
    assert audit == {
        "schema": 1,
        "source_manifest_sha256": hashlib.sha256(source_before).hexdigest(),
        "mutation_policy": "component_paths_only",
        "components": {
            name: {
                "source_path": paths["source"],
                "container_path": paths["container"],
            }
            for name, paths in module.FP32_QUALIFICATION_COMPONENT_PATHS.items()
        },
    }
    assert json.loads(destination.read_text(encoding="utf-8")) == relocated


def test_fp32_qualification_manifest_rejects_unexpected_source_path(
    tmp_path: Path,
) -> None:
    module = load_module()
    source = tmp_path / "manifest.json"
    manifest = {
        "kind": "exact_public_vllm_extraction",
        "components": {
            name: {"path": paths["source"]}
            for name, paths in module.FP32_QUALIFICATION_COMPONENT_PATHS.items()
        },
    }
    manifest["components"]["nano"]["path"] = "/unexpected/nano"
    source.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected nano component path"):
        module.write_fp32_qualification_manifest(source, tmp_path / "relocated.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest.update({"kind": "production_release"}),
            "exact_public_vllm_extraction",
        ),
        (
            lambda manifest: manifest["components"].pop("eartts"),
            "unexpected components",
        ),
        (
            lambda manifest: manifest["components"].update({"extra": {"path": "/work/fp32/extra"}}),
            "unexpected components",
        ),
    ],
)
def test_fp32_qualification_manifest_rejects_wrong_kind_or_component_set(
    tmp_path: Path, mutation, message: str
) -> None:
    module = load_module()
    source = tmp_path / "manifest.json"
    manifest = {
        "kind": "exact_public_vllm_extraction",
        "components": {
            name: {"path": paths["source"]}
            for name, paths in module.FP32_QUALIFICATION_COMPONENT_PATHS.items()
        },
    }
    mutation(manifest)
    source.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        module.write_fp32_qualification_manifest(source, tmp_path / "relocated.json")


def test_direct_semantic_baseline_command_uses_one_checked_in_corpus(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        python=tmp_path / "python",
        model_port=9123,
        direct_semantic_corpus=tmp_path / "corpus.json",
        output=tmp_path / "output",
    )

    command = module.direct_semantic_baseline_command(args)

    assert command[command.index("--url") + 1] == "ws://127.0.0.1:9123/v1/realtime"
    assert command[command.index("--corpus") + 1] == str(args.direct_semantic_corpus)
    assert command[command.index("--output") + 1].endswith("direct-text-semantics/pocket")
    assert "--evidence-only" in command


def test_direct_semantic_image_preflight_is_offline_and_imports_probe() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"

    command = module.direct_semantic_image_preflight_command(image_id)

    assert command[:4] == ["docker", "run", "--rm", "--network"]
    assert "none" in command
    assert command[command.index("--entrypoint") + 1] == "python3"
    assert image_id in command
    assert "nemotron_voicechat_runtime.direct_semantic_probe" in command[-1]


def _complete_direct_probe_report(
    module, *, precision: str, image_id: str, structurally_red: bool
) -> dict:
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    cases = []
    sotc = {
        "pre_eou_suppressed_tokens": 0,
        "pre_eou_last_raw_token_id": None,
        "pre_eou_suppressed_frame": None,
        "client_eou_sotc_committed_count": 0,
        "client_eou_sotc_committed_frame": None,
        "post_fc_client_bos_pending": False,
        "post_fc_client_bos_requested_frame": None,
        "post_fc_client_bos_forced_count": 0,
        "post_fc_client_bos_forced_frame": None,
    }
    for index, case in enumerate(corpus["cases"]):
        structural_passed = not (structurally_red and index == 6)
        structural_checks = {check: True for check in module.DIRECT_SEMANTIC_STRUCTURAL_CHECKS}
        structural_checks["response_complete"] = structural_passed
        settlement_step = {
            "model_step": 1,
            "frame_index": index + 1,
            "audio_frame_index_after": 1,
            "perception_frame_idx_after": 1,
            "eartts_reset_count_after": 0,
            "rnnt_blank_frames": 1,
            "agent_control": "pad",
            "response_boundary": {"event": None, "phase": "idle"},
            "response_audio_deliverable": False,
            "assistant_delta": "",
            "assistant_text_unchanged": True,
            "function_delta": "",
            "function_text_unchanged": True,
            "function_cycle_effective": False,
            "function_calling": {
                "active": False,
                "awaiting_response": False,
                "injecting_response": False,
                "forced_tokens": 0,
                "background_active": False,
                **sotc,
            },
            "vllm_request_positions": {"nano": {"session_positions": index + 1}},
            "violations": [],
        }
        settlement = {
            "target_blank_frames": 1,
            "starting_blank_frames": 0,
            "ending_blank_frames": 1,
            "max_model_steps": 2,
            "model_steps": 1,
            "eartts_reset_count_starting": 0,
            "fence_reached": True,
            "within_bound": True,
            "pre_eou_clean": True,
            "steps": [settlement_step],
        }
        cases.append(
            {
                "case_id": case["id"],
                "semantic": {"passed": index != 2},
                "structural_passed": structural_passed,
                "structural_checks": structural_checks,
                "token_ids": [index + 1],
                "direct_positions": [],
                "direct_state": {
                    "perception_frame_idx_before": 0,
                    "perception_frame_idx_after": 0,
                    "audio_frame_index_after": 0,
                    "eartts_reset_count_before": 0,
                    "eartts_reset_count_after": 0,
                },
                "settlement": settlement,
                "settlement_failure": None,
                "first_settlement_pcm": settlement_step,
                "eou_boundary": {
                    "performed": True,
                    "audio_frame_index_before": 1,
                    "audio_frame_index_after": 2,
                    "perception_frame_idx_before": 1,
                    "perception_frame_idx_after": 2,
                    "frame_index": index + 2,
                    "eartts_reset_count_after": 1,
                    "agent_control": "agent_bos",
                    "response_boundary": {"event": "start", "phase": "responding"},
                    "vllm_request_positions": {"nano": {"session_positions": index + 2}},
                    "sotc_before": copy.deepcopy(sotc),
                    "sotc_after": copy.deepcopy(sotc),
                    "sotc_final": copy.deepcopy(sotc),
                },
                "assistant_text": "",
                "function_text": "",
                "audio_samples": 1,
                "tool_calls": [],
            }
        )
    lanes = [
        {
            "token_form": token_form,
            "case_count": len(cases),
            "cases": copy.deepcopy(cases),
            "structural_passed": not structurally_red,
            "semantic_pass_count": len(cases) - 1,
        }
        for token_form in module.DIRECT_SEMANTIC_TOKEN_FORMS
    ]
    return {
        "schema": 2,
        "kind": "direct_text_semantic_probe",
        "precision": precision,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "runtime_provenance": {"runtime_image_id": image_id},
        "token_forms": list(module.DIRECT_SEMANTIC_TOKEN_FORMS),
        "lanes": lanes,
        "structural_passed": not structurally_red,
        "passed": not structurally_red,
    }


def _with_function_logit_trace(module, report: dict, *, top_k: int = 2) -> dict:
    watched_ids = {"pad": 12, "sotc": 20, "eotc": 21, "eotr": 22, "text_bos": 1}
    report["schema"] = 3
    report["function_logit_trace"] = {
        "top_k": top_k,
        "post_bos_frames": 8,
        "phases": list(module.DIRECT_SEMANTIC_FUNCTION_TRACE_PHASES),
        "watched_token_ids": watched_ids,
    }
    for lane in report["lanes"]:
        for case_index, case in enumerate(lane["cases"]):
            phases = [
                ("direct_position", 0),
                ("settlement", 0),
                ("eou_boundary", 0),
                ("post_bos", 0),
            ]
            entries = []
            for offset, (phase, phase_position) in enumerate(phases):
                top_ids = [12, 20][:top_k]
                entries.append(
                    {
                        "top_k_requested": top_k,
                        "vocab_size": 128,
                        "head_argmax_id": 12,
                        "head_argmax_logit": 5.0,
                        "head_argmax_equal_count": 1,
                        "predicted_token_id": 12,
                        "argmax_matches_predicted": True,
                        "sotc_rank": 2,
                        "sotc_equal_count": 1,
                        "watched": {
                            name: {
                                "token_id": token_id,
                                "logit": float(5 - index),
                            }
                            for index, (name, token_id) in enumerate(watched_ids.items())
                        },
                        "top_k": [
                            {"token_id": token_id, "logit": float(5 - index)}
                            for index, token_id in enumerate(top_ids)
                        ],
                        "model_frame": case_index * 100 + offset,
                        "raw_function_token_id": 12,
                        "effective_function_token_id": 12,
                        "effective_token_reason": "model",
                        "phase": phase,
                        "phase_position": phase_position,
                    }
                )
            case["function_logit_trace"] = entries
    return report


def test_direct_semantic_probe_stage_keeps_complete_red_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report_path = tmp_path / "report.json"
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=True
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, "run", run)
    payload, returncode = module.run_direct_semantic_probe_stage(
        ["probe"],
        log=tmp_path / "probe.log",
        report_path=report_path,
        precision="fp32",
        runtime_image_id=image_id,
        corpus_path=corpus_path,
        timeout=60,
    )

    assert returncode == 1
    assert payload["passed"] is False
    assert calls[0][1]["check"] is False

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )
    with pytest.raises(RuntimeError, match="exited zero with a red report"):
        module.run_direct_semantic_probe_stage(
            ["probe"],
            log=tmp_path / "probe-contradiction.log",
            report_path=report_path,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
            timeout=60,
        )

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 137),
    )
    with pytest.raises(RuntimeError, match="exited unexpectedly: 137"):
        module.run_direct_semantic_probe_stage(
            ["probe"],
            log=tmp_path / "probe-crashed.log",
            report_path=report_path,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
            timeout=60,
        )


def test_direct_semantic_probe_report_rejects_incomplete_red_lane() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=True
    )
    report["lanes"][0]["cases"].pop()

    with pytest.raises(RuntimeError, match="case order/set is incomplete"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_rejects_partial_case_evidence() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=True
    )
    del report["lanes"][0]["cases"][0]["structural_checks"]["clocks_advance_once"]

    with pytest.raises(RuntimeError, match="incomplete structural checks"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_rejects_case_verdict_contradiction() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=True
    )
    report["lanes"][0]["cases"][6]["structural_passed"] = True

    with pytest.raises(RuntimeError, match="inconsistent structural verdict"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_rejects_bare_error() -> None:
    module = load_module()
    with pytest.raises(RuntimeError, match="reported an error"):
        module.validate_direct_semantic_probe_report(
            {
                "schema": 2,
                "kind": "direct_text_semantic_probe",
                "passed": False,
                "error": {"type": "RuntimeError", "message": "preflight failed"},
            },
            precision="fp32",
            runtime_image_id=f"sha256:{'a' * 64}",
            corpus_path=Path("tools/qualification/direct_text_semantic_corpus.json"),
        )


def test_direct_semantic_probe_report_rejects_historical_schema_one() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=False
    )
    report["schema"] = 1

    with pytest.raises(RuntimeError, match="invalid identity"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_accepts_explicit_single_form_diagnostic() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="w8", image_id=image_id, structurally_red=True
    )
    report["token_forms"] = ["text"]
    report["lanes"] = report["lanes"][:1]

    module.validate_direct_semantic_probe_report(
        report,
        precision="w8",
        runtime_image_id=image_id,
        corpus_path=corpus_path,
        expected_token_forms=("text",),
    )


def test_direct_semantic_probe_report_rejects_hidden_settlement_guard_violation() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=False
    )
    report["lanes"][0]["cases"][0]["settlement"]["steps"][0]["agent_control"] = "agent_bos"

    with pytest.raises(RuntimeError, match="contradictory settlement guard evidence"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_rejects_contradictory_settlement_counts() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=False
    )
    report["lanes"][0]["cases"][0]["settlement"]["ending_blank_frames"] = 2

    with pytest.raises(RuntimeError, match="contradictory settlement blank counts"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_rejects_boundary_bootstrap_contradiction() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=False
    )
    report["lanes"][0]["cases"][0]["eou_boundary"]["audio_frame_index_after"] = 3

    with pytest.raises(RuntimeError, match="contradictory settlement structural checks"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_rejects_hidden_direct_eartts_reset() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="fp32", image_id=image_id, structurally_red=False
    )
    report["lanes"][0]["cases"][0]["direct_state"]["eartts_reset_count_after"] = 1

    with pytest.raises(RuntimeError, match="contradictory settlement structural checks"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="fp32",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
        )


def test_direct_semantic_probe_report_accepts_function_logit_trace() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _with_function_logit_trace(
        module,
        _complete_direct_probe_report(
            module, precision="w8", image_id=image_id, structurally_red=False
        ),
    )

    module.validate_direct_semantic_probe_report(
        report,
        precision="w8",
        runtime_image_id=image_id,
        corpus_path=corpus_path,
        expected_trace_top_k=2,
    )


def test_direct_semantic_probe_report_accepts_boundary_sotc_commit_reason() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _with_function_logit_trace(
        module,
        _complete_direct_probe_report(
            module, precision="w8", image_id=image_id, structurally_red=False
        ),
    )
    for lane in report["lanes"]:
        for case in lane["cases"]:
            entry = next(
                value for value in case["function_logit_trace"] if value["phase"] == "eou_boundary"
            )
            entry.update(
                {
                    "head_argmax_id": 20,
                    "predicted_token_id": 20,
                    "raw_function_token_id": 20,
                    "effective_function_token_id": 20,
                    "effective_token_reason": "boundary_sotc_commit",
                    "sotc_rank": 1,
                    "top_k": [
                        {"token_id": 20, "logit": 5.0},
                        {"token_id": 12, "logit": 4.0},
                    ],
                }
            )
            entry["watched"]["sotc"]["logit"] = 5.0
            entry["watched"]["pad"]["logit"] = 4.0

    module.validate_direct_semantic_probe_report(
        report,
        precision="w8",
        runtime_image_id=image_id,
        corpus_path=corpus_path,
        expected_trace_top_k=2,
    )


def test_direct_semantic_probe_report_accepts_sotc_tie_at_top_k_boundary() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _with_function_logit_trace(
        module,
        _complete_direct_probe_report(
            module, precision="w8", image_id=image_id, structurally_red=False
        ),
    )
    for lane in report["lanes"]:
        for case in lane["cases"]:
            for entry in case["function_logit_trace"]:
                entry["sotc_rank"] = 1
                entry["sotc_equal_count"] = 2
                entry["head_argmax_equal_count"] = 2
                entry["watched"]["sotc"]["logit"] = 5.0
                entry["top_k"][1]["logit"] = 5.0

    module.validate_direct_semantic_probe_report(
        report,
        precision="w8",
        runtime_image_id=image_id,
        corpus_path=corpus_path,
        expected_trace_top_k=2,
    )


def test_direct_semantic_probe_report_accepts_argmax_later_in_top_k_tie() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _with_function_logit_trace(
        module,
        _complete_direct_probe_report(
            module, precision="w8", image_id=image_id, structurally_red=False
        ),
    )
    for lane in report["lanes"]:
        for case in lane["cases"]:
            for entry in case["function_logit_trace"]:
                entry["sotc_rank"] = 1
                entry["sotc_equal_count"] = 2
                entry["head_argmax_equal_count"] = 2
                entry["watched"]["sotc"]["logit"] = 5.0
                entry["top_k"] = [
                    {"token_id": 20, "logit": 5.0},
                    {"token_id": 12, "logit": 5.0},
                ]

    module.validate_direct_semantic_probe_report(
        report,
        precision="w8",
        runtime_image_id=image_id,
        corpus_path=corpus_path,
        expected_trace_top_k=2,
    )


def test_direct_semantic_probe_report_accepts_early_end_before_post_bos_bound() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _with_function_logit_trace(
        module,
        _complete_direct_probe_report(
            module, precision="w8", image_id=image_id, structurally_red=False
        ),
    )
    for lane in report["lanes"]:
        for case in lane["cases"]:
            assert case["function_logit_trace"].pop()["phase"] == "post_bos"

    module.validate_direct_semantic_probe_report(
        report,
        precision="w8",
        runtime_image_id=image_id,
        corpus_path=corpus_path,
        expected_trace_top_k=2,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda entry: entry.__setitem__("predicted_token_id", 20),
            "greedy function-head output",
        ),
        (
            lambda entry: entry["watched"]["sotc"].__setitem__("logit", float("nan")),
            "invalid watched function logits",
        ),
        (
            lambda entry: entry.__setitem__("phase_position", 7),
            "ambiguous function-logit phase coverage",
        ),
        (
            lambda entry: entry.__setitem__("effective_token_reason", "pre_eou_guard"),
            "unexplained raw/effective function tokens",
        ),
        (
            lambda entry: entry.__setitem__("model_frame", entry["model_frame"] + 2),
            "duplicate or discontinuous trace frames",
        ),
    ],
)
def test_direct_semantic_probe_report_rejects_function_logit_contradictions(
    mutate, message: str
) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _with_function_logit_trace(
        module,
        _complete_direct_probe_report(
            module, precision="w8", image_id=image_id, structurally_red=False
        ),
    )
    mutate(report["lanes"][0]["cases"][0]["function_logit_trace"][0])

    with pytest.raises(RuntimeError, match=message):
        module.validate_direct_semantic_probe_report(
            report,
            precision="w8",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
            expected_trace_top_k=2,
        )


def test_direct_semantic_probe_report_requires_trace_when_requested() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    report = _complete_direct_probe_report(
        module, precision="w8", image_id=image_id, structurally_red=False
    )

    with pytest.raises(RuntimeError, match="invalid identity"):
        module.validate_direct_semantic_probe_report(
            report,
            precision="w8",
            runtime_image_id=image_id,
            corpus_path=corpus_path,
            expected_trace_top_k=2,
        )


def _set_report_audio_samples(report: dict, samples: int) -> None:
    for lane in report["lanes"]:
        for case in lane["cases"]:
            case["audio_samples"] = samples


def test_function_logit_trace_neutrality_uses_aba_null_and_recursive_strip() -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    control_a = _complete_direct_probe_report(
        module, precision="w8", image_id=image_id, structurally_red=False
    )
    control_b = copy.deepcopy(control_a)
    traced = _with_function_logit_trace(module, copy.deepcopy(control_a))
    _set_report_audio_samples(control_a, module.EARTTS_FRAME_SAMPLES)
    _set_report_audio_samples(traced, module.EARTTS_FRAME_SAMPLES * 2)
    _set_report_audio_samples(control_b, module.EARTTS_FRAME_SAMPLES * 2)
    for report, prefix in ((control_a, "a"), (traced, "trace"), (control_b, "b")):
        for lane in report["lanes"]:
            for case in lane["cases"]:
                case["audio_path"] = f"/{prefix}/{case['case_id']}.wav"
    for report in (control_a, traced, control_b):
        target = report["lanes"][0]["cases"][0]
        target["settlement"]["steps"][0]["model"] = {"raw_function_token_id": 12}
        target["first_settlement_pcm"]["model"] = {"raw_function_token_id": 12}
    for target in (
        traced["lanes"][0]["cases"][0]["settlement"]["steps"][0],
        traced["lanes"][0]["cases"][0]["first_settlement_pcm"],
    ):
        target["model"]["function_logit_trace"] = copy.deepcopy(
            traced["lanes"][0]["cases"][0]["function_logit_trace"][:1]
        )
    control_a["lanes"][0]["cases"][0]["prompt"] = "control-a"
    control_b["lanes"][0]["cases"][0]["prompt"] = "control-b"
    traced["lanes"][0]["cases"][0]["prompt"] = "control-b"

    evidence = module.validate_function_logit_trace_neutrality(
        control_a, traced, control_b
    )

    assert evidence["passed"] is True
    assert evidence["control_nondeterministic_field_count"] == 1
    assert evidence["control_nondeterministic_fields"][0]["within_observed_null"] is True
    assert evidence["c014"]["settlement_and_eou_sotc_rank_two_pad_argmax"] is True


@pytest.mark.parametrize("mode", ["stable", "outside_null", "audio"])
def test_function_logit_trace_neutrality_rejects_unexplained_divergence(mode: str) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    control_a = _complete_direct_probe_report(
        module, precision="w8", image_id=image_id, structurally_red=False
    )
    control_b = copy.deepcopy(control_a)
    traced = _with_function_logit_trace(module, copy.deepcopy(control_a))
    for report in (control_a, traced, control_b):
        _set_report_audio_samples(report, module.EARTTS_FRAME_SAMPLES)
    target_a = control_a["lanes"][0]["cases"][0]
    target_trace = traced["lanes"][0]["cases"][0]
    target_b = control_b["lanes"][0]["cases"][0]
    if mode == "stable":
        target_trace["assistant_text"] = "changed"
    elif mode == "outside_null":
        target_a["prompt"] = "a"
        target_b["prompt"] = "b"
        target_trace["prompt"] = "outside"
    else:
        target_b["audio_samples"] = module.EARTTS_FRAME_SAMPLES * 2
        target_trace["audio_samples"] = module.EARTTS_FRAME_SAMPLES * 3

    evidence = module.validate_function_logit_trace_neutrality(control_a, traced, control_b)

    assert evidence["passed"] is False
    assert evidence["violations"]


def test_function_logit_diagnostic_orchestrates_aba_runs(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus_path = Path("tools/qualification/direct_text_semantic_corpus.json")
    args = argparse.Namespace(
        output=tmp_path / "diagnostic",
        direct_semantic_corpus=corpus_path,
        runtime_image="runtime:test",
        function_logit_trace_top_k=2,
    )
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(module, "runtime_image_id", lambda _image: image_id)
    monkeypatch.setattr(module, "run_checked", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "direct_semantic_probe_command",
        lambda _args, **kwargs: ["probe", str(kwargs["function_logit_trace_top_k"])],
    )

    def stage(_command, **kwargs):
        top_k = kwargs["expected_trace_top_k"]
        calls.append((kwargs["report_path"].parent.name, top_k))
        report = _complete_direct_probe_report(
            module, precision="w8", image_id=image_id, structurally_red=False
        )
        report["token_forms"] = ["text"]
        report["lanes"] = report["lanes"][:1]
        _set_report_audio_samples(report, module.EARTTS_FRAME_SAMPLES)
        if top_k:
            report = _with_function_logit_trace(module, report, top_k=top_k)
        return report, 0

    monkeypatch.setattr(module, "run_direct_semantic_probe_stage", stage)

    result = module.run_direct_function_logit_diagnostic(args)

    assert result["passed"] is True
    assert result["trace_neutrality_within_observed_null"] is True
    assert result["neutrality"]["c014"]["discrete_evidence_equal"] is True
    assert calls == [("off-a", 0), ("on", 2), ("off-b", 0)]


def test_direct_semantic_comparison_stage_keeps_complete_red_report(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    report_path = tmp_path / "comparison.json"
    report = {
        "schema": 1,
        "kind": "direct_text_semantic_comparison",
        "passed": False,
        "corpus_identity_passed": True,
        "runtime_provenance_passed": True,
        "source_provenance_consistent": True,
        "structural_all_forms_passed": False,
        "pocket": {"evidence_complete": True},
        "qualified_token_forms": [],
        "selected_token_form": None,
        "lanes": {
            f"{precision}/{token_form}": {
                "structural_passed": False,
                "hidden_transaction_rederived_passed": True,
                "direct_intent_predicate_passed": False,
                "tool_name_and_arguments_passed": False,
                "safety_critical_passed": True,
                "passed": False,
            }
            for precision in ("fp32", "w8")
            for token_form in module.DIRECT_SEMANTIC_TOKEN_FORMS
        },
        "token_forms": {
            token_form: {
                "fp32_lane_passed": False,
                "w8_lane_passed": False,
                "w8_regression_gate_passed": True,
                "qualified": False,
            }
            for token_form in module.DIRECT_SEMANTIC_TOKEN_FORMS
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1),
    )

    payload, returncode = module.run_direct_semantic_comparison_stage(
        ["compare"],
        log=tmp_path / "comparison.log",
        report_path=report_path,
        timeout=60,
    )

    assert returncode == 1
    assert payload["passed"] is False

    contradictory = copy.deepcopy(report)
    contradictory["lanes"]["fp32/text"]["passed"] = True
    report_path.write_text(json.dumps(contradictory), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lane fp32/text verdict is inconsistent"):
        module.run_direct_semantic_comparison_stage(
            ["compare"],
            log=tmp_path / "comparison-lane-contradiction.log",
            report_path=report_path,
            timeout=60,
        )

    contradictory = copy.deepcopy(report)
    contradictory["token_forms"]["text"]["qualified"] = True
    report_path.write_text(json.dumps(contradictory), encoding="utf-8")
    with pytest.raises(RuntimeError, match="text qualification is inconsistent"):
        module.run_direct_semantic_comparison_stage(
            ["compare"],
            log=tmp_path / "comparison-form-contradiction.log",
            report_path=report_path,
            timeout=60,
        )

    contradictory = copy.deepcopy(report)
    contradictory["structural_all_forms_passed"] = True
    report_path.write_text(json.dumps(contradictory), encoding="utf-8")
    with pytest.raises(RuntimeError, match="structural verdict is inconsistent"):
        module.run_direct_semantic_comparison_stage(
            ["compare"],
            log=tmp_path / "comparison-structural-contradiction.log",
            report_path=report_path,
            timeout=60,
        )


def test_direct_semantic_ladder_rejects_pocket_image_id_mismatch(tmp_path: Path) -> None:
    module = load_module()
    root = tmp_path / "output/direct-text-semantics"
    (root / "pocket").mkdir(parents=True)
    (root / "pocket/report.json").write_text(
        json.dumps({"runtime_provenance": {"runtime_image_id": f"sha256:{'b' * 64}"}}),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        output=tmp_path / "output",
        direct_semantic_corpus=corpus,
        runtime_image="runtime:test",
        fp32_derived_root=tmp_path / "fp32",
    )

    with pytest.raises(RuntimeError, match="Pocket report image ID mismatch"):
        module.run_direct_semantic_gpu_ladder(args, image_id=f"sha256:{'a' * 64}")


def test_direct_semantic_ladder_rejects_probe_image_id_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    expected_image_id = f"sha256:{'a' * 64}"
    root = tmp_path / "output/direct-text-semantics"
    (root / "pocket").mkdir(parents=True)
    (root / "pocket/report.json").write_text(
        json.dumps({"runtime_provenance": {"runtime_image_id": expected_image_id}}),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        output=tmp_path / "output",
        direct_semantic_corpus=corpus,
        runtime_image="runtime:test",
        fp32_derived_root=tmp_path / "fp32",
    )
    monkeypatch.setattr(
        module,
        "direct_semantic_probe_command",
        lambda _args, *, precision, runtime_image_id: ["probe", precision],
    )
    monkeypatch.setattr(
        module,
        "write_fp32_qualification_manifest",
        lambda _source, _destination: {},
    )

    monkeypatch.setattr(
        module,
        "run_direct_semantic_probe_stage",
        lambda _command, **_kwargs: (
            {"runtime_provenance": {"runtime_image_id": f"sha256:{'b' * 64}"}},
            0,
        ),
    )

    with pytest.raises(RuntimeError, match="fp32 report image ID mismatch"):
        module.run_direct_semantic_gpu_ladder(args, image_id=expected_image_id)


def test_direct_semantic_ladder_continues_fp32_red_through_w8_and_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    image_id = f"sha256:{'a' * 64}"
    corpus = Path("tools/qualification/direct_text_semantic_corpus.json")
    root = tmp_path / "output/direct-text-semantics"
    (root / "pocket").mkdir(parents=True)
    (root / "pocket/report.json").write_text(
        json.dumps({"runtime_provenance": {"runtime_image_id": image_id}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        output=tmp_path / "output",
        direct_semantic_corpus=corpus,
        runtime_image="runtime:test",
        fp32_derived_root=tmp_path / "fp32",
        python=tmp_path / "python",
    )
    calls = []
    monkeypatch.setattr(
        module,
        "write_fp32_qualification_manifest",
        lambda _source, _destination: {},
    )
    monkeypatch.setattr(
        module,
        "direct_semantic_probe_command",
        lambda _args, *, precision, runtime_image_id: [
            "probe",
            precision,
            runtime_image_id,
        ],
    )

    def probe_stage(_command, **kwargs):
        precision = kwargs["precision"]
        calls.append(precision)
        report = _complete_direct_probe_report(
            module,
            precision=precision,
            image_id=image_id,
            structurally_red=True,
        )
        kwargs["report_path"].write_text(json.dumps(report), encoding="utf-8")
        return report, 1

    monkeypatch.setattr(module, "run_direct_semantic_probe_stage", probe_stage)

    def comparison_stage(_command, **kwargs):
        calls.append("compare")
        report = {
            "schema": 1,
            "kind": "direct_text_semantic_comparison",
            "passed": False,
        }
        kwargs["report_path"].write_text(json.dumps(report), encoding="utf-8")
        return report, 1

    monkeypatch.setattr(module, "run_direct_semantic_comparison_stage", comparison_stage)

    report = module.run_direct_semantic_gpu_ladder(args, image_id=image_id)
    state = json.loads((root / "ladder.json").read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert calls == ["fp32", "w8", "compare"]
    assert [stage["status"] for stage in state["stages"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert state["passed"] is False
    assert state["stages"][1]["container_returncode"] == 1
    assert state["stages"][2]["container_returncode"] == 1
    assert state["stages"][3]["evaluator_returncode"] == 1


def test_direct_semantics_only_orders_pocket_stop_fp32_w8_ladder(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    fp32 = tmp_path / "fp32"
    (fp32 / "nano-vllm-fp32").mkdir(parents=True)
    (fp32 / "eartts-vllm-fp32").mkdir()
    for path in (
        fp32 / "manifest.json",
        fp32 / "nano-vllm-fp32/config.json",
        fp32 / "eartts-vllm-fp32/config.json",
    ):
        path.write_text("{}", encoding="utf-8")
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}", encoding="utf-8")
    order = []

    class FakeStack:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            order.append("stack-stop")
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(module, "load_config", lambda: {"image": {"runtime": "runtime:test"}})
    monkeypatch.setattr(module, "runtime_image_id", lambda _image: f"sha256:{'a' * 64}")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: FakeStack())
    monkeypatch.setattr(
        module, "wait_for_playground", lambda *_args, **_kwargs: order.append("ready")
    )
    monkeypatch.setattr(module, "wait_model_idle", lambda *_args: order.append("idle"))

    def checked(command, **_kwargs):
        order.append("preflight" if "--entrypoint" in command else "pocket")

    monkeypatch.setattr(module, "run_checked", checked)

    def ladder(_args, *, image_id):
        assert image_id == f"sha256:{'a' * 64}"
        order.append("fp32-w8-compare")
        return {"passed": True}

    monkeypatch.setattr(module, "run_direct_semantic_gpu_ladder", ladder)
    args = argparse.Namespace(
        fp32_derived_root=fp32,
        direct_semantic_corpus=corpus,
        runtime_image="runtime:test",
        output=tmp_path / "output",
        python=tmp_path / "python",
        cache_root=tmp_path / "cache",
        trace_root=tmp_path / "traces",
        model_port=8786,
        pipecat_port=7860,
        startup_timeout=60,
    )

    report = module.run_direct_semantics_only(args)

    assert report["passed"]
    assert order == [
        "preflight",
        "ready",
        "idle",
        "pocket",
        "idle",
        "stack-stop",
        "fp32-w8-compare",
    ]


def test_conversion_tensor_tests_run_offline_in_runtime_image(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    commands = []
    args = argparse.Namespace(output=tmp_path, runtime_image="runtime:test")

    def capture(command, **kwargs):
        commands.append((command, kwargs))

    monkeypatch.setattr(module, "run_checked", capture)
    module.run_conversion_tensor_tests(args)
    command, kwargs = commands[0]
    assert command[:6] == ["docker", "run", "--rm", "--network", "none", "--workdir"]
    assert "--gpus" not in command
    assert command[-6:] == [
        "tests/conversion/test_nano_attribution.py",
        "tests/conversion/test_nano_gptq_calibration.py",
        "tests/conversion/test_package_replay_corpus.py",
        "tests/conversion/test_nano_replay.py",
        "tests/conversion/test_eartts_component_gate.py",
        (
            "tests/runtime/test_pocket_worker_ipc.py::"
            "test_internal_text_carrier_is_deterministic_per_text_and_restores_rng"
        ),
    ]
    assert kwargs["log"] == tmp_path / "conversion-tensor-tests.log"


def test_gpu_component_prefix_uses_production_tokenizer_mount(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        cache_root=tmp_path / "cache",
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
        asr_evaluator_image_id=f"sha256:{'a' * 64}",
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
            "memory_matrix": {
                "cases": [
                    {
                        "seed_audio_path": "/qualification/fixtures/memory-seed.wav",
                        "recall_audio_path": "/qualification/fixtures/turn-03.wav",
                    }
                ]
            },
            "browser_microphone_wav": "/qualification/fixtures/browser-microphone.wav",
        }
        (fixtures / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module, "run_checked", generate)
    module.prepare_fixtures(args)
    assert args.fixture_manifest == output / "fixture-manifest-host.json"
    manifest = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    assert manifest["turns"][0]["audio_path"] == str(output / "fixtures/turn-01.wav")
    assert manifest["memory_matrix"]["cases"][0]["seed_audio_path"] == str(
        output / "fixtures/memory-seed.wav"
    )
    assert manifest["memory_matrix"]["cases"][0]["recall_audio_path"] == str(
        output / "fixtures/turn-03.wav"
    )
    assert args.browser_mic_wav == output / "fixtures/browser-microphone.wav"


def test_browser_gate_records_failure_then_success(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = argparse.Namespace(python=Path("/runtime/python"), output=tmp_path, model_port=8786)
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
    args = argparse.Namespace(python=Path("/runtime/python"), output=tmp_path, model_port=8786)

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
