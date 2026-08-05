#!/usr/bin/env python3
"""Apply the qualified, weight-preserving Nano runtime configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from nano_gptq_calibration import sha256_file

CONVERTER_CONFIG_SHA256 = "9179d7190c3708000cb9fdfa9495b7fded63259384b35185547d1fa42e6100bd"
QUALIFIED_PREDECESSOR_CONFIG_SHA256 = (
    "caf9fe075b3baae60d39990bc6a48d48413857b7dd5f321350b3e236cc6681b9"
)
QUALIFIED_CONFIG_SHA256 = "2a0576a8b71aa09a5a6683d5a0d2e3fc099839ea9fe559d69a3458e828530391"
MINIMUM_FREE_BYTES = 100 * 1024**3


def qualified_config(source_config: dict) -> dict:
    if source_config.get("custom_outputs") != [
        "text_logits",
        "function_tokens",
        "function_logits",
    ]:
        raise ValueError("converter Nano config has an unexpected custom-output contract")
    result = dict(source_config)
    result["chunk_size"] = 1
    result["custom_outputs"] = ["function_tokens", "function_logits"]
    result["voicechat_skip_custom_text_logits"] = {
        "contract": "temperature=0, top_p=1, repetition_penalty=1",
        "enabled": True,
        "source_config_sha256": CONVERTER_CONFIG_SHA256,
    }
    result["voicechat_pad_pair_token_id"] = 12
    result["voicechat_pad_pair_contract"] = {
        "agent_decoded_silence_frames": 20,
        "agent_no_audio_frames": 30,
        "agent_no_text_frames": 30,
        "conditional_previous_effective_pad": True,
        "function_async_pending_drain": True,
        "function_async_sequential_latch": True,
        "function_channel_rejection": True,
        "mamba_chunk_size": 1,
        "max_positions": 2,
        "max_session_model_frames": 12000,
        "post_fc_fusion_correction": True,
        "production_candidate": 1,
        "production_mamba_rollback": True,
        "scope": "production live runtime",
        # This is the exact config-only predecessor of the artifact that passed
        # the live gates. Retaining it makes the release config byte-identical.
        "source_config_sha256": QUALIFIED_PREDECESSOR_CONFIG_SHA256,
        "streaming_max_len": 12288,
        "vllm_max_model_len": 12288,
    }
    return result


def finalize(source_dir: Path, output_dir: Path) -> dict:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    source_config_path = source_dir / "config.json"
    if sha256_file(source_config_path) != CONVERTER_CONFIG_SHA256:
        raise ValueError("Nano converter config does not match the qualified recipe input")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < MINIMUM_FREE_BYTES:
        raise RuntimeError("Refusing to run below the 100 GiB free-space floor")

    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    copy_modes: dict[str, str] = {}
    try:
        temporary.mkdir()
        for source in sorted(source_dir.iterdir()):
            if not source.is_file():
                raise ValueError(f"unexpected non-file in Nano artifact: {source}")
            if source.name == "config.json":
                continue
            destination = temporary / source.name
            try:
                os.link(source, destination)
                copy_modes[source.name] = "hardlink"
            except OSError:
                shutil.copy2(source, destination)
                copy_modes[source.name] = "copy"
        config = qualified_config(json.loads(source_config_path.read_text(encoding="utf-8")))
        output_config = temporary / "config.json"
        output_config.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        actual = sha256_file(output_config)
        if actual != QUALIFIED_CONFIG_SHA256:
            raise RuntimeError(
                f"qualified Nano config digest drift: {actual} != {QUALIFIED_CONFIG_SHA256}"
            )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        "config_sha256": QUALIFIED_CONFIG_SHA256,
        "weights_changed": False,
        "copy_modes": copy_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.source_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
