#!/usr/bin/env python3
"""Add a hash-qualified FULL CUDA graph for the exact Nano PAD-pair shape."""

from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path


PURELIB = Path("/usr/local/lib/python3.12/dist-packages")
TARGETS = {
    "forward_context": (
        PURELIB / "vllm/forward_context.py",
        "d288bda150d32e2f15d3a27ba66276e112d26c43d3b44861796e052d24bb11b5",
    ),
    "dispatcher": (
        PURELIB / "vllm/v1/cudagraph_dispatcher.py",
        "3ab7a422cfcf4cc876c58f44a3ded2bc41863cc03567c7859f6f22860d72316f",
    ),
    "runner": (
        PURELIB / "vllm/v1/worker/gpu_model_runner.py",
        "2ce42b6fcafa1ff3320b78efe1d9c36462e646982c47ab088eed60d47605925f",
    ),
    "mamba_attn": (
        PURELIB / "vllm/v1/attention/backends/mamba_attn.py",
        "a622099edf787fc0ab693d6a8a68547b6c8707e879309f5e1013977767ad6d3d",
    ),
    "runtime": (
        PURELIB / "nemotron_voicechat_runtime/runtime_optimizations.py",
        "ca89643197e5facbea1b61698d044ea7b2ad89be94886c14103017ff42509c34",
    ),
}
MARKER = "voicechat_step7_pair_full_graph_v1"


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def qualified_source(name: str) -> tuple[Path, str]:
    path, expected = TARGETS[name]
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"unqualified input {path}: {actual} != {expected}")
    return path, data.decode("utf-8")


def patch_forward_context(source: str) -> str:
    source = replace_once(
        source,
        """    num_tokens: int
    uniform_decode: bool = False
""",
        f"""    num_tokens: int
    uniform_decode: bool = False
    # {MARKER}: distinguish the one-request/two-token PAD pair from the
    # ordinary two-request/single-token uniform graph with the same token count.
    query_len: int = 0
""",
        "BatchDescriptor query length",
    )
    return replace_once(
        source,
        """        return BatchDescriptor(self.num_tokens, uniform_decode=False)
""",
        """        return BatchDescriptor(
            self.num_tokens, uniform_decode=False, query_len=self.query_len
        )
""",
        "BatchDescriptor non-uniform projection",
    )


def patch_dispatcher(source: str) -> str:
    source = replace_once(source, "from typing import Optional\n", "import os\nfrom typing import Optional\n", "dispatcher os import")
    source = replace_once(
        source,
        """        self.keys_initialized = False
""",
        f"""        self.keys_initialized = False
        # {MARKER}: the model config is the semantic opt-in; the environment
        # flag makes the experimental graph independently switchable without
        # changing the frozen release manifest or production configuration.
        self.voicechat_pair_full_graph = bool(
            getattr(
                self.vllm_config.model_config.hf_config,
                "voicechat_pad_pair_token_id",
                None,
            ) is not None
            and os.environ.get("VOICECHAT_NANO_PAIR_FULL_GRAPH", "0").strip().lower()
            in {{"1", "true", "yes", "on"}}
        )
""",
        "dispatcher pair opt-in",
    )
    return replace_once(
        source,
        """        self.keys_initialized = True
""",
        f"""        if (
            self.voicechat_pair_full_graph
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        ):
            # {MARKER}: this key is intentionally distinct from the normal
            # two-request decode graph. Only the configured one-PAD proposal
            # constructs query_len=2 at runtime.
            self.add_cudagraph_key(
                CUDAGraphMode.FULL,
                BatchDescriptor(num_tokens=2, uniform_decode=True, query_len=2),
            )
        self.keys_initialized = True
""",
        "dispatcher pair key",
    )


def patch_mamba_attn(source: str) -> str:
    source = replace_once(source, "import abc\n", "import abc\nimport os\n", "mamba os import")
    return replace_once(
        source,
        """        assert m.num_reqs == m.num_actual_tokens, (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        m.max_query_len = 1  # decode-only

        return self.build(0, m)
""",
        f"""        # {MARKER}: the production pair is a fixed one-request,
        # two-token initialized continuation. It must retain the existing
        # chunk-size-one prefill scan and intermediate-state rollback path;
        # treating it as two independent decode requests would be incorrect.
        pair_capture = bool(
            m.num_reqs == 1
            and m.num_actual_tokens == 2
            and getattr(
                self.vllm_config.model_config.hf_config,
                "voicechat_pad_pair_token_id",
                None,
            ) is not None
            and os.environ.get("VOICECHAT_NANO_PAIR_FULL_GRAPH", "0").strip().lower()
            in {{"1", "true", "yes", "on"}}
        )
        if pair_capture:
            if self.chunk_size != 1:
                raise RuntimeError(
                    "Nano PAD-pair FULL graph requires Mamba chunk_size=1"
                )
            return self.build(0, m)

        assert m.num_reqs == m.num_actual_tokens, (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        m.max_query_len = 1  # decode-only

        return self.build(0, m)
""",
        "Mamba pair capture metadata",
    )


def patch_runner(source: str) -> str:
    source = replace_once(source, "import gc\n", "import gc\nimport json\nimport os\n", "runner telemetry imports")
    source = replace_once(
        source,
        """        self.uniform_decode_query_len = (
            1
            if not self.speculative_config
            else 1 + self.speculative_config.num_speculative_tokens
        )
""",
        f"""        self.uniform_decode_query_len = (
            1
            if not self.speculative_config
            else 1 + self.speculative_config.num_speculative_tokens
        )
        # {MARKER}: this is separate from speculative_config because the
        # custom-input scheduler supplies exactly one configured PAD proposal.
        self.voicechat_pair_full_graph = bool(
            getattr(
                self.model_config.hf_config,
                "voicechat_pad_pair_token_id",
                None,
            ) is not None
            and os.environ.get("VOICECHAT_NANO_PAIR_FULL_GRAPH", "0").strip().lower()
            in {{"1", "true", "yes", "on"}}
        )
""",
        "runner pair opt-in",
    )
    source = replace_once(
        source,
        """    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"):
""",
        f"""    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        # {MARKER}: host monotonic timestamps are process-comparable on this
        # machine. Logging remains fully absent unless the opt-in flag is set.
        voicechat_execute_started_ns = (
            time.perf_counter_ns()
            if os.environ.get("VOICECHAT_PAIR_RPC_PROFILE", "0").strip().lower()
            in {{"1", "true", "yes", "on"}}
            else 0
        )
        with record_function_or_nullcontext("Preprocess"):
""",
        "worker execution timing start",
    )
    source = replace_once(
        source,
        """            uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
                num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
            )
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens, uniform_decode=uniform_decode
            )
""",
        f"""            uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
                num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
            )
            # {MARKER}: require every production invariant that identifies the
            # custom one-PAD proposal. Initial prefill, serial continuation,
            # CFG/function bypass, and any broader shape retain old dispatch.
            voicechat_pair_step = bool(
                spec_decode_metadata is not None
                and self.input_batch.num_reqs == 1
                and max_query_len == 2
                and num_scheduled_tokens == 2
                and num_input_tokens == 2
            )
            voicechat_pair_full_graph = bool(
                self.voicechat_pair_full_graph and voicechat_pair_step
            )
            if (
                voicechat_pair_full_graph
                and os.environ.get("VOICECHAT_STEP7_BITWISE_GATE", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
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
                        f"{{request_id}}"
                    )
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens,
                uniform_decode=(uniform_decode or voicechat_pair_full_graph),
                query_len=2 if voicechat_pair_full_graph else 0,
            )
""",
        "runtime pair dispatch",
    )
    source = replace_once(
        source,
        """        if not self.use_async_scheduling:
            return output
""",
        f"""        if (
            voicechat_pair_step
            and os.environ.get("VOICECHAT_STEP7_BITWISE_GATE", "0")
            .strip()
            .lower()
            in {{"1", "true", "yes", "on"}}
        ):
            # {MARKER}: compare persistent state while the streaming request's
            # row and block table are still live. The scheduler retires that
            # row before a parent-side collective RPC can inspect it.
            request_id = str(self.input_batch.req_ids[0])
            request_index = self.input_batch.req_id_to_index[request_id]
            layer_groups = {{}}
            for group_index, group in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):
                for layer_name in group.layer_names:
                    layer_groups[layer_name] = group_index
            block_ids = {{}}
            for group_index in range(len(self.kv_cache_config.kv_cache_groups)):
                table = self.input_batch.block_table[group_index]
                count = int(table.num_blocks_per_row[request_index])
                block_ids[group_index] = [
                    int(value)
                    for value in table.get_numpy_array()[request_index, :count]
                ]

            current_states = []
            unmatched_layers = []
            for layer_name, layer in self.model.named_modules():
                if not hasattr(layer, "kv_cache"):
                    continue
                group_index = layer_groups.get(layer_name)
                if group_index is None:
                    unmatched_layers.append(layer_name)
                    continue
                stack = [(layer_name, layer.kv_cache)]
                while stack:
                    state_name, state = stack.pop()
                    if isinstance(state, torch.Tensor):
                        current_states.append(
                            (
                                state_name,
                                state[block_ids[group_index]],
                            )
                        )
                    elif isinstance(state, (list, tuple)):
                        for state_index in range(len(state) - 1, -1, -1):
                            stack.append(
                                (f"{{state_name}}[{{state_index}}]", state[state_index])
                            )
            if unmatched_layers:
                raise RuntimeError(
                    "Step-7 cache layers absent from KV groups: %s"
                    % unmatched_layers
                )
            if not current_states:
                raise RuntimeError("Step-7 bitwise gate found no persistent cache state")

            if request_id.startswith("step7-old-"):
                self._voicechat_step7_cache_reference = [
                    (name, value.clone()) for name, value in current_states
                ]
            elif request_id.startswith("step7-new-"):
                reference = getattr(
                    self, "_voicechat_step7_cache_reference", None
                )
                if reference is None:
                    raise RuntimeError(
                        "Step-7 candidate pair ran without a baseline state snapshot"
                    )
                if len(reference) != len(current_states):
                    raise RuntimeError(
                        "Step-7 persistent cache tensor counts differ: %d != %d"
                        % (len(reference), len(current_states))
                    )
                compared_elements = 0
                for (reference_name, reference_value), (
                    current_name,
                    current_value,
                ) in zip(reference, current_states):
                    if (
                        reference_name != current_name
                        or reference_value.dtype != current_value.dtype
                        or reference_value.shape != current_value.shape
                    ):
                        raise RuntimeError(
                            "Step-7 persistent cache metadata differs at %s / %s"
                            % (reference_name, current_name)
                        )
                    compared_elements += current_value.numel()
                    if not torch.equal(reference_value, current_value):
                        differing = int(
                            torch.count_nonzero(reference_value != current_value)
                        )
                        raise RuntimeError(
                            "Step-7 persistent cache mismatch at %s: %d/%d elements"
                            % (current_name, differing, current_value.numel())
                        )
                results = getattr(self, "_voicechat_step7_cache_results", None)
                if results is None:
                    results = []
                    self._voicechat_step7_cache_results = results
                result = {{
                    "equal": True,
                    "sequence": len(results),
                    "compared_tensors": len(current_states),
                    "compared_elements": compared_elements,
                }}
                results.append(result)
                logger.warning(
                    "VOICECHAT_STEP7_CACHE_EQUAL %s",
                    json.dumps(result, sort_keys=True),
                )
                del self._voicechat_step7_cache_reference
            else:
                raise RuntimeError(
                    "Step-7 bitwise gate received an unqualified state request: %s"
                    % request_id
                )

        if voicechat_execute_started_ns and voicechat_pair_step:
            voicechat_execute_finished_ns = time.perf_counter_ns()
            logger.warning(
                "VOICECHAT_PAIR_WORKER %s",
                json.dumps(
                    {{
                        "execute_started_ns": voicechat_execute_started_ns,
                        "execute_finished_ns": voicechat_execute_finished_ns,
                        "execute_ms": round(
                            (voicechat_execute_finished_ns - voicechat_execute_started_ns)
                            / 1e6,
                            6,
                        ),
                        "num_requests": int(self.input_batch.num_reqs),
                        "num_scheduled_tokens": int(num_scheduled_tokens),
                        "max_query_len": int(max_query_len),
                        "cudagraph_mode": cudagraph_runtime_mode.name,
                        "pair_full_graph": voicechat_pair_full_graph,
                    }},
                    sort_keys=True,
                ),
            )

        if not self.use_async_scheduling:
            return output
""",
        "worker execution timing finish",
    )
    source = replace_once(
        source,
        """            torch.cuda.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]
""",
        f"""            if self.voicechat_pair_full_graph:
                # {MARKER}: capture after normal decode graphs so the distinct
                # descriptor cannot collide with the existing 2x1 graph.
                self._capture_cudagraphs(
                    compilation_cases=[2],
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True,
                    query_len_override=2,
                )

            torch.cuda.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]
""",
        "pair full graph capture",
    )
    source = replace_once(
        source,
        """        uniform_decode: bool,
    ):
""",
        """        uniform_decode: bool,
        query_len_override: int = 0,
    ):
""",
        "capture helper query length",
    )
    source = source.replace(
        """                    uniform_decode=uniform_decode,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
""",
        """                    uniform_decode=uniform_decode,
                    query_len_override=query_len_override,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
""",
        1,
    )
    source = source.replace(
        """                uniform_decode=uniform_decode,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
""",
        """                uniform_decode=uniform_decode,
                query_len_override=query_len_override,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
""",
        1,
    )
    source = replace_once(
        source,
        """        uniform_decode: bool = False,
        allow_microbatching: bool = True,
""",
        """        uniform_decode: bool = False,
        query_len_override: int = 0,
        allow_microbatching: bool = True,
""",
        "dummy query length parameter",
    )
    source = replace_once(
        source,
        """        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
""",
        f"""        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
        if query_len_override:
            # {MARKER}: only the dedicated capture call supplies this override.
            if not (uniform_decode and query_len_override == 2 and num_tokens == 2):
                raise RuntimeError("invalid Nano PAD-pair capture shape")
            max_query_len = query_len_override
""",
        "dummy pair max query length",
    )
    source = replace_once(
        source,
        """        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())

        ubatch_slices = None
""",
        f"""        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())
        if query_len_override:
            # {MARKER}: make capture exercise the initialized-continuation
            # branch, including conv/SSM shadow snapshots. Runtime state/index
            # tensors remain persistent and are updated before every replay.
            self.input_batch.num_computed_tokens_cpu[:num_reqs] = 1

        ubatch_slices = None
""",
        "dummy initialized continuation",
    )
    source = replace_once(
        source,
        """                    BatchDescriptor(
                        num_tokens=num_tokens_after_padding,
                        uniform_decode=uniform_decode,
                    )
""",
        """                    BatchDescriptor(
                        num_tokens=num_tokens_after_padding,
                        uniform_decode=uniform_decode,
                        query_len=query_len_override,
                    )
""",
        "dummy pair descriptor",
    )
    return replace_once(
        source,
        """        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]
""",
        """        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        if query_len_override:
            self.input_batch.num_computed_tokens_cpu[:num_reqs] = 0
        return hidden_states, hidden_states[logit_indices]
""",
        "dummy pair state cleanup",
    )


def patch_runtime(source: str) -> str:
    source = replace_once(source, "import inspect\n", "import inspect\nimport json\n", "runtime json import")
    source = replace_once(source, "import os\n", "import os\nimport time\n", "runtime time import")
    source = replace_once(
        source,
        """    await engine.engine.append_request(
        request_id=request_id,
        custom_inputs=custom_inputs,
    )
    output = await request_state.generation_iterator.__anext__()
""",
        f"""    profile_rpc = enabled("VOICECHAT_PAIR_RPC_PROFILE")
    append_started_ns = time.perf_counter_ns() if profile_rpc else 0
    await engine.engine.append_request(
        request_id=request_id,
        custom_inputs=custom_inputs,
    )
    append_finished_ns = time.perf_counter_ns() if profile_rpc else 0
    wait_started_ns = append_finished_ns
    output = await request_state.generation_iterator.__anext__()
    wait_finished_ns = time.perf_counter_ns() if profile_rpc else 0
    if profile_rpc:
        # {MARKER}: this parent-side span includes EngineCore scheduling,
        # worker execution, IPC, and output assembly. A paired worker span is
        # logged by gpu_model_runner for residual-RPC subtraction.
        logging.warning(
            "VOICECHAT_PAIR_RPC %s",
            json.dumps(
                {{
                    "request_id": str(request_id),
                    "generated_before": len(request_state.generated_tokens),
                    "append_started_ns": append_started_ns,
                    "append_finished_ns": append_finished_ns,
                    "wait_started_ns": wait_started_ns,
                    "wait_finished_ns": wait_finished_ns,
                    "append_ms": round((append_finished_ns - append_started_ns) / 1e6, 6),
                    "response_wait_ms": round((wait_finished_ns - wait_started_ns) / 1e6, 6),
                }},
                sort_keys=True,
            ),
        )
""",
        "parent pair RPC timing",
    )
    return source


def main() -> None:
    patchers = {
        "forward_context": patch_forward_context,
        "dispatcher": patch_dispatcher,
        "runner": patch_runner,
        "mamba_attn": patch_mamba_attn,
        "runtime": patch_runtime,
    }
    for name, patcher in patchers.items():
        path, source = qualified_source(name)
        patched = patcher(source)
        if MARKER not in patched:
            raise RuntimeError(f"{name}: marker missing after patch")
        path.write_text(patched, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)
        print(
            f"patched {name}: {path} "
            f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
        )


if __name__ == "__main__":
    main()
