#!/usr/bin/env python3
"""Avoid an unused duplicate text-head GEMV in deterministic public inference."""

from __future__ import annotations

import hashlib
import py_compile
import sysconfig
from pathlib import Path

MARKER = "voicechat_public_skip_unused_custom_text_logits"


def main() -> None:
    path = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/models/nemotron_h.py"
    source = path.read_text(encoding="utf-8")
    before = hashlib.sha256(source.encode()).hexdigest()
    if MARKER in source:
        print(f"Already patched: {path} sha256={before}")
        return

    anchor = "        text_logits = self.compute_logits(hidden_states)\n"
    replacement = f"""        # {MARKER}: vLLM's ordinary sampler already evaluates
        # lm_head after forward. Deterministic VoiceChat consumes that sampled token,
        # so a converted config may omit custom text_logits and avoid evaluating the
        # same 131072 x 4480 head a second time inside the CUDA graph.
        text_logits = (
            self.compute_logits(hidden_states)
            if "text_logits" in self.custom_outputs
            else None
        )
"""
    count = source.count(anchor)
    if count != 1:
        raise SystemExit(f"Expected one custom text-logit anchor, found {count}")
    source = source.replace(anchor, replacement, 1)
    path.write_text(source, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    after = hashlib.sha256(source.encode()).hexdigest()
    print(f"Patched: {path} before={before} after={after}")


if __name__ == "__main__":
    main()
