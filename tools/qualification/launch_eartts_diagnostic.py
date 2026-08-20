#!/usr/bin/env python3
"""Launch a separately named, bind-mounted EarTTS diagnostic model server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from nemotron_voicechat_runtime.artifacts import default_layout, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def command(args: argparse.Namespace) -> list[str]:
    config = load_config(args.config)
    layout = default_layout()
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.image],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    environment = dict(config["runtime"]["environment"])
    environment.update(
        {
            "HF_HOME": "/models/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/models/huggingface/hub",
            "HF_MODULES_CACHE": "/tmp/voicechat-hf-modules",
            "VOICECHAT_RUNTIME_CONTRACT": config["candidate"],
            "VOICECHAT_RUNTIME_IMAGE": args.image,
            "VOICECHAT_RUNTIME_IMAGE_ID": image_id,
        }
    )
    if args.nano_memory_utilization is not None:
        environment["VOICECHAT_VLLM_NANO_MEMORY_UTILIZATION"] = str(
            args.nano_memory_utilization
        )
    if args.eartts_memory_utilization is not None:
        environment["VOICECHAT_VLLM_EARTTS_MEMORY_UTILIZATION"] = str(
            args.eartts_memory_utilization
        )
    if args.disable_prepared_epoch:
        environment["VOICECHAT_EARTTS_PREPARED_EPOCH"] = "0"
    if args.capture_root is not None:
        environment.update(
            {
                "S2S_EARTTS_ACOUSTIC_CAPTURE_DIR": "/capture",
                "S2S_EARTTS_ACOUSTIC_CAPTURE_WRITER_CPU": str(args.writer_cpu),
            }
        )
        if args.silence_frames is not None:
            environment["VOICECHAT_DIAGNOSTIC_AGENT_SILENCE_EOS_FRAMES"] = str(
                args.silence_frames
            )
        if args.no_progress_frames is not None:
            environment["VOICECHAT_DIAGNOSTIC_AGENT_NO_PROGRESS_FRAMES"] = str(
                args.no_progress_frames
            )
    if args.max_agent_response_seconds is not None:
        environment["VOICECHAT_DIAGNOSTIC_MAX_AGENT_RESPONSE_SEC"] = str(
            args.max_agent_response_seconds
        )
    if args.lock_step:
        environment["VOICECHAT_DIAGNOSTIC_LOCKSTEP"] = "1"
    trace_root = args.trace_root.expanduser().resolve()
    trace_root.mkdir(parents=True, exist_ok=False)
    nano_container_path = "/models/candidates/nano-fhw8"
    eartts_container_path = "/models/derived/eartts"
    manifest_container_path = "/models/candidates/fhw8-candidate-manifest.json"
    launch = [
        "docker",
        "run",
        "--detach",
        "--name",
        args.name,
        "--init",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--security-opt",
        "label=disable",
        "--label",
        "ai.pipecat.voicechat.experiment=eartts-truncation-diagnostic-r1",
        "-p",
        f"127.0.0.1:{args.port}:8786",
        "-v",
        f"{layout.artifacts / 'candidates'}:/models/candidates:ro",
        "-v",
        f"{layout.parent}:/models/voicechat:ro",
        "-v",
        f"{layout.nano_skeleton}:/models/NVIDIA-Nemotron-Nano-9B-v2:ro",
        "-v",
        f"{layout.release}:/models/derived:ro",
        "-v",
        f"{layout.hf_home}:/models/huggingface:ro",
        "-v",
        f"{trace_root}:/trace",
        "-v",
        (
            f"{REPO_ROOT / 'src/nemotron_voicechat_runtime'}:"
            "/usr/local/lib/python3.12/dist-packages/nemotron_voicechat_runtime:ro"
        ),
        "-v",
        f"{REPO_ROOT / 'config'}:/usr/local/lib/python3.12/config:ro",
    ]
    alternate_manifest = None
    if args.nano_path is not None or args.eartts_path is not None:
        alternate_manifest = json.loads(
            (layout.artifacts / "candidates/fhw8-candidate-manifest.json").read_text(
                encoding="utf-8"
            )
        )
    if args.nano_path is not None:
        nano_path = args.nano_path.expanduser().resolve()
        required = (nano_path / "config.json", nano_path / "model.safetensors")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                f"diagnostic Nano artifact is incomplete: {nano_path}"
            )
        environment["VOICECHAT_DIAGNOSTIC_NANO_CONFIG_SHA256"] = hashlib.sha256(
            required[0].read_bytes()
        ).hexdigest()
        nano_container_path = "/models/nano-diagnostic"
        launch.extend(["-v", f"{nano_path}:{nano_container_path}:ro"])
        if args.nano_manifest is None:
            raise ValueError("alternate Nano artifact requires --nano-manifest")
        source_manifest_path = args.nano_manifest.expanduser().resolve()
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        component = copy.deepcopy((source_manifest.get("components") or {}).get("nano"))
        if not component:
            raise ValueError(f"manifest has no Nano component: {source_manifest_path}")
        component["path"] = nano_container_path
        declared_config = (component.get("files") or {}).get("config.json") or {}
        if declared_config.get("sha256") != environment[
            "VOICECHAT_DIAGNOSTIC_NANO_CONFIG_SHA256"
        ]:
            raise ValueError("alternate Nano config differs from its source manifest")
        assert alternate_manifest is not None
        alternate_manifest["components"]["nano"] = component
    if args.eartts_path is not None:
        eartts_path = args.eartts_path.expanduser().resolve()
        required = (eartts_path / "config.json", eartts_path / "model.safetensors")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                f"diagnostic EarTTS artifact is incomplete: {eartts_path}"
            )
        environment["VOICECHAT_DIAGNOSTIC_EARTTS_CONFIG_SHA256"] = hashlib.sha256(
            required[0].read_bytes()
        ).hexdigest()
        environment["VOICECHAT_DIAGNOSTIC_EARTTS_AB"] = "1"
        eartts_container_path = "/models/eartts-diagnostic"
        launch.extend(["-v", f"{eartts_path}:{eartts_container_path}:ro"])
        if args.eartts_manifest is None:
            raise ValueError("alternate EarTTS artifact requires --eartts-manifest")
        source_manifest_path = args.eartts_manifest.expanduser().resolve()
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        component = copy.deepcopy((source_manifest.get("components") or {}).get("eartts"))
        if not component:
            raise ValueError(f"manifest has no EarTTS component: {source_manifest_path}")
        component["path"] = eartts_container_path
        declared_config = (component.get("files") or {}).get("config.json") or {}
        if declared_config.get("sha256") != environment[
            "VOICECHAT_DIAGNOSTIC_EARTTS_CONFIG_SHA256"
        ]:
            raise ValueError("alternate EarTTS config differs from its source manifest")
        assert alternate_manifest is not None
        alternate_manifest["components"]["eartts"] = component
    if alternate_manifest is not None:
        if args.nano_path is not None and args.eartts_path is not None:
            alternate_manifest["kind"] = "exact_public_vllm_full_precision_ab"
            environment["VOICECHAT_DIAGNOSTIC_FULL_PRECISION_AB"] = "1"
        elif args.eartts_path is not None:
            alternate_manifest["kind"] = "exact_public_vllm_eartts_quantization_ab"
            environment["VOICECHAT_DIAGNOSTIC_EARTTS_AB"] = "1"
        else:
            raise ValueError("Nano-only precision diagnostic is not implemented")
        diagnostic_manifest = trace_root / "diagnostic-vllm-manifest.json"
        diagnostic_manifest.write_text(
            json.dumps(alternate_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_container_path = "/trace/diagnostic-vllm-manifest.json"
    if args.speech_wrapper is not None:
        wrapper = args.speech_wrapper.expanduser().resolve()
        if not wrapper.is_file():
            raise FileNotFoundError(wrapper)
        environment["VOICECHAT_DIAGNOSTIC_SPEECH_WRAPPER_SHA256"] = hashlib.sha256(
            wrapper.read_bytes()
        ).hexdigest()
        launch.extend(
            [
                "-v",
                (
                    f"{wrapper}:/opt/Speech/nemo/collections/speechlm2/inference/"
                    "model_wrappers/nemotron_voicechat_inference_wrapper.py:ro"
                ),
            ]
        )
    if args.capture_root is not None:
        capture_root = args.capture_root.expanduser().resolve()
        capture_root.mkdir(parents=True, exist_ok=False)
        launch.extend(["-v", f"{capture_root}:/capture"])
    for name, value in sorted(environment.items()):
        launch.extend(["-e", f"{name}={value}"])
    launch.extend(
        [
            args.image,
            "python3",
            "-m",
            "nemotron_voicechat_runtime.server",
            "--speech-root",
            "/opt/Speech",
            "--checkpoint-root",
            "/models/voicechat",
            "--hf-skeleton",
            "/models/NVIDIA-Nemotron-Nano-9B-v2",
            "--vllm-manifest",
            manifest_container_path,
            "--nano-vllm-path",
            nano_container_path,
            "--eartts-vllm-path",
            eartts_container_path,
            "--speaker-name",
            "Aria",
            "--host",
            "0.0.0.0",
            "--port",
            "8786",
            "--trace-dir",
            "/trace",
        ]
    )
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/production-hotfix-notext-watchdog-v1.toml",
    )
    parser.add_argument(
        "--image",
        default="pipecat-ai/nemotron-voicechat-dgx-spark:session-repair-20260818-r4",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--speech-wrapper", type=Path)
    parser.add_argument(
        "--nano-path",
        type=Path,
        help="bind an alternate immutable Nano artifact into the diagnostic container",
    )
    parser.add_argument(
        "--nano-manifest",
        type=Path,
        help="source manifest attesting an alternate Nano artifact",
    )
    parser.add_argument(
        "--eartts-path",
        type=Path,
        help="bind an alternate immutable EarTTS artifact into the diagnostic container",
    )
    parser.add_argument(
        "--eartts-manifest",
        type=Path,
        help="source manifest attesting an alternate EarTTS artifact",
    )
    parser.add_argument("--silence-frames", type=int)
    parser.add_argument("--no-progress-frames", type=int)
    parser.add_argument("--max-agent-response-seconds", type=float)
    parser.add_argument("--lock-step", action="store_true")
    parser.add_argument("--writer-cpu", type=int, default=10)
    parser.add_argument("--nano-memory-utilization", type=float)
    parser.add_argument("--eartts-memory-utilization", type=float)
    parser.add_argument(
        "--disable-prepared-epoch",
        action="store_true",
        help=(
            "force the diagnostic onto the genuine reset-on-BOS path by disabling "
            "prepared-epoch reuse"
        ),
    )
    args = parser.parse_args()
    if args.name == "nemotron-voicechat-model" or args.port == 8786:
        raise SystemExit("diagnostic must use a renamed container and a non-production port")
    if args.capture_root is None and args.silence_frames is not None:
        raise SystemExit("silence threshold can change only with acoustic capture enabled")
    full_precision_ab = args.nano_path is not None and args.eartts_path is not None
    if args.capture_root is None and not full_precision_ab and (
        args.max_agent_response_seconds is not None or args.lock_step
    ):
        raise SystemExit(
            "diagnostic timing controls require acoustic capture or the attested "
            "full-precision A/B"
        )
    if args.nano_manifest is not None and args.nano_path is None:
        raise SystemExit("--nano-manifest requires --nano-path")
    if args.eartts_manifest is not None and args.eartts_path is None:
        raise SystemExit("--eartts-manifest requires --eartts-path")
    completed = subprocess.run(command(args), check=True, text=True, capture_output=True)
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
