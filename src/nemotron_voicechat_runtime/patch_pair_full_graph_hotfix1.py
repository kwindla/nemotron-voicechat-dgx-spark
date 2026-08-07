#!/usr/bin/env python3
"""Incrementally repair the Step-7 cache gate in the surviving overlay image."""

from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")
EXPECTED_SHA256 = "e6d2b9fe98267cd5abbf3e53e1070629e0d69f253d35d2eacaca811016409d79"
OLD_REBIND = """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("Postprocess"):
"""
NEW_REBIND = """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        if voicechat_pair_full_graph:
            # voicechat_step7_pair_full_graph_v1: CUDA replay updates the
            # tensor shadows but cannot replay the Python attribute assignment
            # that points rollback at the current request's Mamba block.
            if not isinstance(attn_metadata, dict):
                raise RuntimeError("Nano PAD-pair FULL graph requires dict metadata")
            rebound_layers = 0
            for module_name, module in self.model.named_modules():
                if not callable(
                    getattr(module, "voicechat_restore_pad_pair_state", None)
                ):
                    continue
                module_metadata = attn_metadata.get(module_name)
                if module_metadata is None:
                    raise RuntimeError(
                        "Nano PAD-pair FULL graph lacks metadata for %s" % module_name
                    )
                module.voicechat_pad_pair_shadow_index = (
                    module_metadata.state_indices_tensor.reshape(-1)[:1].clone()
                )
                rebound_layers += 1
            if rebound_layers == 0:
                raise RuntimeError("Nano PAD-pair FULL graph rebound no Mamba layers")

        with record_function_or_nullcontext("Postprocess"):
"""
OLD_SHADOW = """            current_states = []
            unmatched_layers = []
"""
NEW_SHADOW = """            current_states = []
            # The rejection path consumes graph-written first-position
            # shadows after model replay. Compare those graph side effects as
            # well as the committed cache so a rollback failure is localized.
            for module_name, module in self.model.named_modules():
                for shadow_name in (
                    "voicechat_pad_pair_shadow_conv",
                    "voicechat_pad_pair_shadow_ssm",
                ):
                    shadow = getattr(module, shadow_name, None)
                    if isinstance(shadow, torch.Tensor) and shadow.numel():
                        current_states.append(
                            (f"{module_name}.{shadow_name}", shadow)
                        )
            unmatched_layers = []
"""
OLD = """                    if isinstance(state, torch.Tensor):
                        current_states.append(
                            (
                                state_name,
                                state[block_ids[group_index]],
                            )
                        )
"""
NEW = """                    if isinstance(state, torch.Tensor):
                        if state.numel() == 0:
                            continue
                        selected_block_ids = block_ids[group_index]
                        maximum_block_id = max(selected_block_ids)
                        if (
                            state.ndim > 1
                            and state.shape[0] in (1, 2)
                            and maximum_block_id < state.shape[1]
                        ):
                            # Hybrid attention buffers can retain the legacy
                            # [K/V, block, ...] layout while Mamba state uses
                            # [block, ...]. Compare the request blocks on the
                            # actual block axis in either representation.
                            selected_state = state[:, selected_block_ids]
                        elif maximum_block_id < state.shape[0]:
                            selected_state = state[selected_block_ids]
                        else:
                            raise RuntimeError(
                                "Step-7 cannot locate cache block axis for %s: "
                                "shape=%s max_block_id=%d"
                                % (state_name, tuple(state.shape), maximum_block_id)
                            )
                        current_states.append(
                            (
                                state_name,
                                selected_state,
                            )
                        )
"""
OLD_DISPATCH = """            if (
                voicechat_pair_full_graph
                and os.environ.get("VOICECHAT_STEP7_BITWISE_GATE", "0")
                .strip()
                .lower()
                in ('1', 'true', 'yes', 'on')
            ):
                # The exactness harness keeps two independent requests in one
                # worker process.  Only its explicitly named baseline request
                # is held on the unmodified PIECEWISE path.
                request_id = str(self.input_batch.req_ids[0])
                if request_id.startswith("step7-old-"):
                    voicechat_pair_full_graph = False
                elif not request_id.startswith("step7-new-"):
                    raise RuntimeError(
                        "Step-7 bitwise gate received an unqualified pair request: "
                        f"{request_id}"
                    )
"""
NEW_DISPATCH = """            voicechat_step7_bitwise_gate = (
                os.environ.get("VOICECHAT_STEP7_BITWISE_GATE", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            voicechat_step7_perf_side_by_side = (
                os.environ.get("VOICECHAT_STEP7_PERF_SIDE_BY_SIDE", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            if voicechat_step7_bitwise_gate and voicechat_step7_perf_side_by_side:
                raise RuntimeError(
                    "Step-7 bitwise and performance routing are mutually exclusive"
                )
            if voicechat_pair_full_graph and voicechat_step7_bitwise_gate:
                # The exactness harness keeps two independent requests in one
                # worker process.  Only its explicitly named baseline request
                # is held on the unmodified PIECEWISE path.
                request_id = str(self.input_batch.req_ids[0])
                if request_id.startswith("step7-old-"):
                    voicechat_pair_full_graph = False
                elif not request_id.startswith("step7-new-"):
                    raise RuntimeError(
                        "Step-7 bitwise gate received an unqualified pair request: "
                        f"{request_id}"
                    )
            elif voicechat_pair_full_graph and voicechat_step7_perf_side_by_side:
                # Alternate modes inside one worker so engine and RNG state
                # remain directly comparable: even IDs PIECEWISE, odd FULL.
                request_id = str(self.input_batch.req_ids[0])
                try:
                    request_number = int(request_id)
                except ValueError as exc:
                    raise RuntimeError(
                        "Step-7 performance routing requires a numeric request ID: "
                        f"{request_id}"
                    ) from exc
                voicechat_pair_full_graph = bool(request_number % 2)
"""


def main() -> None:
    source = TARGET.read_text()
    actual = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"unqualified input {TARGET}: {actual} != {EXPECTED_SHA256}")
    if source.count(OLD) != 1:
        raise RuntimeError(f"expected one cache-selection anchor, found {source.count(OLD)}")
    if source.count(OLD_SHADOW) != 1:
        raise RuntimeError(f"expected one shadow anchor, found {source.count(OLD_SHADOW)}")
    if source.count(OLD_REBIND) != 1:
        raise RuntimeError(f"expected one rebind anchor, found {source.count(OLD_REBIND)}")
    if source.count(OLD_DISPATCH) != 1:
        raise RuntimeError(f"expected one dispatch anchor, found {source.count(OLD_DISPATCH)}")
    source = source.replace(OLD_DISPATCH, NEW_DISPATCH, 1)
    source = source.replace(OLD_REBIND, NEW_REBIND, 1)
    source = source.replace(OLD_SHADOW, NEW_SHADOW, 1)
    TARGET.write_text(source.replace(OLD, NEW, 1))
    py_compile.compile(str(TARGET), doraise=True)
    print(
        "patched Step-7 verification and performance routing: "
        f"sha256={hashlib.sha256(TARGET.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
