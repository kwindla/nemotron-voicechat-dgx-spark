#!/usr/bin/env python3
"""Build the preregistered, tokenizer-exact tool-result payload manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

TARGETS = (20, 48, 72, 120)
VERIFICATION_WORDS = ("River", "Meadow", "Harbor", "Cedar")
FILLER_WORDS = (
    "is",
    "the",
    "fresh",
    "verification",
    "word",
    "for",
    "this",
    "request",
    "please",
    "remember",
    "and",
    "repeat",
    "it",
    "clearly",
    "in",
    "your",
    "answer",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def wrapped(output: str) -> str:
    return f"<TOOL_RESPONSE>[{output}]</TOOL_RESPONSE>"


def build_payload(tokenizer: Any, target: int, word: str) -> dict[str, Any]:
    words = [word]
    filler_index = 0
    while True:
        output = " ".join(words) + "."
        serialized = wrapped(output)
        token_ids = list(tokenizer.encode(serialized, add_special_tokens=False))
        if len(token_ids) >= target:
            if len(token_ids) > 128:
                raise RuntimeError(f"target {target} first exceeded at {len(token_ids)} tokens")
            concise_output = f"The fresh word is {word}."
            concise_wrapped = wrapped(concise_output)
            concise_token_ids = list(
                tokenizer.encode(concise_wrapped, add_special_tokens=False)
            )
            if len(concise_token_ids) > 20:
                raise RuntimeError(
                    f"concise output for {word} uses {len(concise_token_ids)} tokens"
                )
            return {
                "target_tokens": target,
                "realized_tokens": len(token_ids),
                "verification_word": word,
                "output": output,
                "output_utf8_bytes": len(output.encode()),
                "wrapped": serialized,
                "wrapped_sha256": sha256_bytes(serialized.encode()),
                "token_ids": token_ids,
                "model_output": concise_output,
                "model_output_utf8_bytes": len(concise_output.encode()),
                "model_output_wrapped": concise_wrapped,
                "model_output_wrapped_sha256": sha256_bytes(concise_wrapped.encode()),
                "model_output_tokens": len(concise_token_ids),
                "model_output_token_ids": concise_token_ids,
            }
        words.append(FILLER_WORDS[filler_index % len(FILLER_WORDS)])
        filler_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer_files = {
        name: sha256_bytes((args.tokenizer / name).read_bytes())
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    payloads = [
        build_payload(tokenizer, target, word)
        for target, word in zip(TARGETS, VERIFICATION_WORDS, strict=True)
    ]
    if len({row["realized_tokens"] for row in payloads}) != len(payloads):
        raise RuntimeError("payload token counts are not distinct")
    manifest = {
        "schema": 1,
        "kind": "tool_call_latency_payloads",
        "tokenizer_files": tokenizer_files,
        "limits": {"max_utf8_bytes": 16 * 1024, "max_wrapped_tokens": 128},
        "payloads": payloads,
    }
    manifest["sha256"] = sha256_bytes(canonical_bytes(manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
