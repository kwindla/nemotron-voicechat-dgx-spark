#!/usr/bin/env python3
"""Pad the exact VoiceChat K=15680 GPTQ group-128 projection to K=15744."""

from __future__ import annotations

import py_compile
import sysconfig
from pathlib import Path

MARKER = "voicechat_marlin_group128_input_padding"


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def main() -> None:
    path = (
        Path(sysconfig.get_paths()["purelib"])
        / "vllm/model_executor/layers/quantization/gptq_marlin.py"
    )
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        """        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
""",
        f"""        # {MARKER}: K=15680 has a 64-element tail at group 128.
        # The serialized W8 candidate pads the tail with exact zeros. Allocate
        # Marlin's parameters/kernel at K=15744 while leaving the model-visible
        # linear shape unchanged; apply() pads matching zero activations.
        voicechat_input_padding = 0
        if (
            self.quant_config.group_size == 128
            and self.quant_config.quant_type.size_bits == 8
            and input_size_per_partition == 15680
            and input_size == 15680
            and sum(output_partition_sizes) == 4480
        ):
            voicechat_input_padding = 64
            input_size_per_partition += voicechat_input_padding
            input_size += voicechat_input_padding
        layer._voicechat_marlin_input_padding = voicechat_input_padding

        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
""",
        "weight allocation padding",
    )
    source = replace_once(
        source,
        """    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
""",
        f"""    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # {MARKER}
        padding = getattr(layer, \"_voicechat_marlin_input_padding\", 0)
        if padding:
            x = torch.nn.functional.pad(x, (0, padding))
        return self.kernel.apply_weights(layer, x, bias)
""",
        "activation padding",
    )
    path.write_text(source, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
