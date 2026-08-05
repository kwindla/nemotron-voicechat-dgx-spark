#!/usr/bin/env python3
"""Install the EA W8A32 quantization registration into vLLM."""

import shutil
from pathlib import Path

source = Path("/tmp/ea_w8a32.py")
package = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization")
target = package / "ea_w8a32.py"
registry = package / "__init__.py"

shutil.copy2(source, target)
bf16_head_source = Path("/tmp/ea_bf16_head.py")
bf16_head_target = package.parent / "ea_bf16_head.py"
shutil.copy2(bf16_head_source, bf16_head_target)
text = registry.read_text()
marker = "# EA W8A32 registration"
if marker not in text:
    methods_anchor = "QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))\n"
    import_anchor = "    from .auto_round import AutoRoundConfig\n"
    mapping_anchor = '        "awq": AWQConfig,\n'
    if not all(anchor in text for anchor in (methods_anchor, import_anchor, mapping_anchor)):
        raise RuntimeError("Unsupported vLLM quantization registry layout")
    text = text.replace(
        methods_anchor,
        methods_anchor
        + "\n# EA W8A32 registration: advertise the method without importing linear.py.\n"
        + 'QUANTIZATION_METHODS.append("ea_w8a32")\n',
        1,
    )
    text = text.replace(
        import_anchor,
        import_anchor + "    from .ea_w8a32 import EAW8A32Config\n",
        1,
    )
    text = text.replace(
        mapping_anchor,
        mapping_anchor + '        "ea_w8a32": EAW8A32Config,\n',
        1,
    )
    registry.write_text(text)
print(f"Installed {target} and registered ea_w8a32")

eartts = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/eartts.py")
eartts_text = eartts.read_text()
sampler_marker = "# EA W8A32 sampler MLP"
if sampler_marker not in eartts_text:
    class_anchor = "class MLP(nn.Module):\n"
    sampler_assignment = "self.sampler = MaskGITSampler(vllm_config.model_config.hf_config)"
    init_lines = [
        line for line in eartts_text.splitlines(keepends=True) if line.strip() == sampler_assignment
    ]
    if class_anchor not in eartts_text or len(init_lines) != 1:
        raise RuntimeError("Unsupported vLLM EarTTS model layout")
    init_anchor = init_lines[0]
    init_indent = init_anchor[: len(init_anchor) - len(init_anchor.lstrip())]
    body_indent = init_indent + (" " if len(init_indent) == 2 else "    ")
    sampler_linear = '''# EA W8A32 sampler MLP
class EAW8A32TorchLinear(nn.Module):
    """FP32-activation/INT8-weight linear for ordinary EarTTS modules."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, 1, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.ea.w8a32_linear.default(
            value, self.weight, self.weight_scale.reshape(-1)
        )


def install_w8a32_sampler_mlp(sampler: nn.Module) -> None:
    for layer in sampler.mog_head.mlp_stack:
        if not isinstance(layer, MLPLayer):
            continue
        mlp = layer.mlp
        mlp.gate_proj = EAW8A32TorchLinear(
            mlp.gate_proj.in_features, mlp.gate_proj.out_features
        )
        mlp.up_proj = EAW8A32TorchLinear(
            mlp.up_proj.in_features, mlp.up_proj.out_features
        )
        mlp.down_proj = EAW8A32TorchLinear(
            mlp.down_proj.in_features, mlp.down_proj.out_features
        )


'''
    eartts_text = eartts_text.replace(class_anchor, sampler_linear + class_anchor, 1)
    eartts_text = eartts_text.replace(
        init_anchor,
        init_anchor
        + init_indent
        + "if getattr(vllm_config.model_config.hf_config, "
        + "'ea_w8a32_sampler_mlp', False):\n"
        + body_indent
        + "install_w8a32_sampler_mlp(self.sampler)\n",
        1,
    )
    eartts.write_text(eartts_text)
    print(f"Installed W8A32 EarTTS sampler MLP support in {eartts}")

nemotron = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/nemotron_h.py")
nemotron_text = nemotron.read_text()
head_marker = "EA_BF16_ASR_GEMV"
if head_marker not in nemotron_text:
    import_anchor = "from collections.abc import Iterable\n"
    logits_anchor = (
        "        self.logits_processor = LogitsProcessor("
        "self.unpadded_vocab_size, config.vocab_size)\n"
    )
    if import_anchor not in nemotron_text or logits_anchor not in nemotron_text:
        raise RuntimeError("Unsupported vLLM Nemotron-H model layout")
    nemotron_text = nemotron_text.replace(import_anchor, import_anchor + "import os\n", 1)
    nemotron_text = nemotron_text.replace(
        logits_anchor,
        "        if os.environ.get('EA_BF16_ASR_GEMV', '0') == '1':\n"
        "            from vllm.model_executor.layers.ea_bf16_head import (\n"
        "                EABF16HeadLinearMethod,\n"
        "            )\n"
        "            self.asr_head.quant_method = EABF16HeadLinearMethod()\n" + logits_anchor,
        1,
    )
    nemotron.write_text(nemotron_text)
    print(f"Installed opt-in BF16 ASR head GEMV support in {nemotron}")
