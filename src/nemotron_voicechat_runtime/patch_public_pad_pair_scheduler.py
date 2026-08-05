#!/usr/bin/env python3
"""Enable an explicitly configured one-PAD draft on custom-input appends.

The public streaming fork already accepts multiple custom-input positions and
vLLM already has exact target-model rejection for a caller-supplied draft
token.  The missing connection is that ``set_custom_inputs`` normally exposes
only the single outstanding decode position to the scheduler.  A model config
with ``voicechat_pad_pair_token_id`` opts into treating every additional custom
input row as that PAD draft token.  The initial prompt/prefill path is unchanged.

This patch is an inner-loop discriminator.  Rejected drafts still require a
hybrid Mamba-state rollback before this may be used in production.
"""

from __future__ import annotations

import hashlib
import py_compile
import sysconfig
from pathlib import Path

MARKER = "voicechat_public_pad_pair_scheduler"
MAMBA_MARKER = "voicechat_public_pad_pair_mamba_rollback"


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def main() -> None:
    root = Path(sysconfig.get_paths()["purelib"]) / "vllm"
    path = root / "v1/core/sched/scheduler.py"
    source = path.read_text(encoding="utf-8")
    before = hashlib.sha256(source.encode()).hexdigest()
    if MARKER in source:
        print(f"Already patched: {path} sha256={before}")
        return

    source = replace_once(
        source,
        "        self.await_inputs = vllm_config.model_config.custom_input_specs is not None\n",
        f"""        self.await_inputs = vllm_config.model_config.custom_input_specs is not None
        # {MARKER}: opt-in only; ordinary models and single-row appends retain
        # byte-for-byte scheduler behavior.
        self.voicechat_pad_pair_token_id = getattr(
            vllm_config.model_config.hf_config,
            "voicechat_pad_pair_token_id",
            None,
        )
""",
        "scheduler opt-in config",
    )
    source = replace_once(
        source,
        """        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

""",
        f"""        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
        elif self.voicechat_pad_pair_token_id is not None:
            # {MARKER}: size speculative acceptance telemetry for one PAD.
            # No KV lookahead is needed; both slots allocate in this step.
            self.num_spec_tokens = 1

""",
        "PAD-pair speculative telemetry width",
    )
    source = replace_once(
        source,
        """        request.set_custom_inputs(custom_inputs)
        if request_id in self.waiting_input:
""",
        f"""        # {MARKER}: a two-row continuation consists of the ordinary
        # outstanding decode position plus one explicitly assumed PAD draft.
        # vLLM's existing speculative rejection path then evaluates both target
        # logits in one backbone pass and accepts the second only if the first
        # target token equals this draft.
        if self.voicechat_pad_pair_token_id is not None:
            input_lengths = {{int(value.shape[0]) for value in custom_inputs.values()}}
            if len(input_lengths) != 1:
                raise ValueError(
                    f"PAD-pair custom inputs have inconsistent lengths: {{input_lengths}}"
                )
            input_length = input_lengths.pop()
            if input_length > 2:
                raise ValueError(
                    f"PAD-pair scheduler supports at most two positions, got {{input_length}}"
                )
            if input_length == 2:
                # The consumed draft remains in this list until an ordinary
                # proposer replaces it. This opt-in path is itself the
                # proposer, so each serialized append replaces the stale PAD.
                request.spec_token_ids = [int(self.voicechat_pad_pair_token_id)]
            else:
                # A non-drafted step must also replace the previous proposal;
                # otherwise the stale PAD makes the scheduler request two rows.
                request.spec_token_ids = []

        request.set_custom_inputs(custom_inputs)
        if request_id in self.waiting_input:
""",
        "custom-input PAD draft",
    )
    source = replace_once(
        source,
        """            # Share the same custom_inputs reference for CFG pair
            cfg_uncond.set_custom_inputs(custom_inputs)
""",
        f"""            # Share the same custom_inputs reference for CFG pair.
            # {MARKER}: mirror an explicitly installed PAD draft as well.
            cfg_uncond.spec_token_ids = request.spec_token_ids.copy()
            cfg_uncond.set_custom_inputs(custom_inputs)
""",
        "CFG PAD draft mirror",
    )

    path.write_text(source, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    after = hashlib.sha256(source.encode()).hexdigest()
    print(f"Patched: {path} before={before} after={after}")

    runner_path = root / "v1/worker/gpu_model_runner.py"
    runner_source = runner_path.read_text(encoding="utf-8")
    runner_before = hashlib.sha256(runner_source.encode()).hexdigest()
    runner_source = replace_once(
        runner_source,
        """            self.rejection_sampler = RejectionSampler()

        # Request states.
""",
        f"""            self.rejection_sampler = RejectionSampler()

        # {MARKER}: the PAD-pair target uses vLLM's rejection kernel without
        # configuring a separate draft model/proposer.
        if (
            not self.speculative_config
            and getattr(
                self.model_config.hf_config,
                "voicechat_pad_pair_token_id",
                None,
            ) is not None
            and get_pp_group().is_last_rank
        ):
            self.rejection_sampler = RejectionSampler()

        # Request states.
""",
        "PAD-pair rejection sampler",
    )
    runner_source = replace_once(
        runner_source,
        """        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
""",
        f"""        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        # {MARKER}: custom-embedding positions do not carry meaningful token
        # IDs in input_ids.gpu. The scheduler's sole configured proposal is
        # PAD, so install that exact ID in rejection metadata rather than
        # accepting an unrelated stale buffer value.
        pad_pair_token_id = getattr(
            self.model_config.hf_config,
            "voicechat_pad_pair_token_id",
            None,
        )
        if pad_pair_token_id is not None:
            draft_token_ids = torch.full_like(
                draft_token_ids,
                int(pad_pair_token_id),
            )

        metadata = SpecDecodeMetadata(
""",
        "custom-input PAD rejection metadata",
    )
    runner_source = replace_once(
        runner_source,
        """            sampler_output.sampled_token_ids = output_token_ids
            self._update_states_after_model_execute(output_token_ids)
""",
        f"""            # {MARKER}: position one assumes PAD on every recurrent
            # agent channel. Text rejection is handled above by vLLM's native
            # sampler; apply the same condition to the always-on function head
            # before committing bookkeeping or Mamba state.
            pad_pair_token_id = getattr(
                self.model_config.hf_config,
                "voicechat_pad_pair_token_id",
                None,
            )
            if (
                pad_pair_token_id is not None
                and output_token_ids.shape == (1, 2)
                and custom_outputs_flat is not None
            ):
                function_tokens = custom_outputs_flat.get("function_tokens")
                if function_tokens is None or function_tokens.numel() < 2:
                    raise RuntimeError(
                        "PAD-pair acceptance requires per-position function_tokens"
                    )
                function_rejected = function_tokens.reshape(-1)[0].ne(
                    int(pad_pair_token_id)
                )
                output_token_ids[0, 1] = torch.where(
                    function_rejected,
                    torch.full_like(output_token_ids[0, 1], -1),
                    output_token_ids[0, 1],
                )

            sampler_output.sampled_token_ids = output_token_ids

            # {MAMBA_MARKER}: a rejected one-PAD draft leaves only the first
            # target token in column zero and -1 in column one. Mamba prefill
            # has already committed the second position, so restore the exact
            # chunk-1 state retained by every Mamba2 layer. Accepted pairs do
            # no cache copy and retain the fast common path.
            if getattr(
                self.model_config.hf_config,
                "voicechat_pad_pair_token_id",
                None,
            ) is not None:
                if output_token_ids.shape != (1, 2):
                    raise RuntimeError(
                        "PAD-pair rollback currently requires one request and "
                        f"one draft, got {{tuple(output_token_ids.shape)}}"
                    )
                draft_rejected = bool(output_token_ids[0, 1].eq(-1).item())
                if draft_rejected:
                    restored_layers = sum(
                        int(module.voicechat_restore_pad_pair_state())
                        for module in self.get_model().modules()
                        if callable(
                            getattr(module, "voicechat_restore_pad_pair_state", None)
                        )
                    )
                    if restored_layers == 0:
                        raise RuntimeError(
                            "PAD draft was rejected but no Mamba2 shadow state "
                            "was available"
                        )

            self._update_states_after_model_execute(output_token_ids)
""",
        "rejected PAD Mamba rollback",
    )
    runner_source = replace_once(
        runner_source,
        """    def _sample(
        self,
        logits: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
    ) -> SamplerOutput:    
""",
        f"""    def _sample(
        self,
        logits: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        custom_outputs_flat: Optional[dict[str, torch.Tensor]] = None,
    ) -> SamplerOutput:
        # {MARKER}: custom outputs are needed only to make the one-PAD draft
        # acceptance conjunctive across text and function channels.
""",
        "function-aware sampler signature",
    )
    runner_source = replace_once(
        runner_source,
        """            sampler_output = self._sample(logits, spec_decode_metadata)
""",
        f"""            # {MARKER}: pass per-position function tokens into the
            # rejection decision before recurrent state/bookkeeping commits.
            sampler_output = self._sample(
                logits,
                spec_decode_metadata,
                custom_outputs_flat,
            )
""",
        "function-aware sampler call",
    )
    runner_path.write_text(runner_source, encoding="utf-8")
    py_compile.compile(str(runner_path), doraise=True)
    runner_after = hashlib.sha256(runner_source.encode()).hexdigest()
    print(f"Patched: {runner_path} before={runner_before} after={runner_after}")

    mamba_path = root / "model_executor/layers/mamba/mamba_mixer2.py"
    mamba_source = mamba_path.read_text(encoding="utf-8")
    mamba_before = hashlib.sha256(mamba_source.encode()).hexdigest()
    mamba_source = replace_once(
        mamba_source,
        """        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
""",
        f"""        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        # {MAMBA_MARKER}: buffers are allocated lazily after KV-cache binding,
        # only for the explicitly configured PAD-pair artifact.
        self.voicechat_pad_pair_enabled = bool(
            model_config is not None
            and getattr(
                model_config.hf_config,
                "voicechat_pad_pair_token_id",
                None,
            ) is not None
        )
        self.voicechat_pad_pair_shadow_conv = None
        self.voicechat_pad_pair_shadow_ssm = None
        self.voicechat_pad_pair_shadow_index = None
        self.voicechat_pad_pair_shadow_virtual_engine = 0
""",
        "Mamba PAD-pair config",
    )
    mamba_source = replace_once(
        mamba_source,
        """            x = hidden_states_B_C_p.transpose(
                0, 1
            )  # this is the form that causal-conv see
            hidden_states_B_C_p = causal_conv1d_fn(
""",
        f"""            x = hidden_states_B_C_p.transpose(
                0, 1
            )  # this is the form that causal-conv see

            # {MAMBA_MARKER}: with chunk size one, the two logical chunks are
            # exactly the states produced by two recurrent single-token calls.
            # Retain the state after position zero before the kernels commit
            # position one. This path is deliberately bounded to one initialized
            # continuation request; larger/mixed batches fail closed.
            pad_pair_active = (
                self.voicechat_pad_pair_enabled
                and num_prefills == 1
                and num_prefill_tokens == 2
                and chunk_size == 1
                and prep_initial_states
            )
            if pad_pair_active:
                if prefix_caching_enabled:
                    raise RuntimeError(
                        "PAD-pair Mamba rollback does not support prefix caching"
                    )
                shadow_index = state_indices_tensor_p.reshape(-1)[:1].clone()
                prior_conv = conv_state[shadow_index].squeeze(0)
                if (
                    self.voicechat_pad_pair_shadow_conv is None
                    or self.voicechat_pad_pair_shadow_conv.shape != prior_conv.shape
                ):
                    self.voicechat_pad_pair_shadow_conv = torch.empty_like(prior_conv)
                self.voicechat_pad_pair_shadow_conv[..., :-1].copy_(
                    prior_conv[..., 1:]
                )
                self.voicechat_pad_pair_shadow_conv[..., -1].copy_(x[:, 0])
                self.voicechat_pad_pair_shadow_index = shadow_index
                self.voicechat_pad_pair_shadow_virtual_engine = (
                    forward_context.virtual_engine
                )

            hidden_states_B_C_p = causal_conv1d_fn(
""",
        "Mamba first-position conv shadow",
    )
    mamba_source = replace_once(
        mamba_source,
        """                return_intermediate_states=prefix_caching_enabled,
""",
        f"""                # {MAMBA_MARKER}: chunk-size-1 pair mode needs the
                # state at both logical chunk boundaries; ordinary execution
                # retains the original final-state-only allocation.
                return_intermediate_states=(
                    prefix_caching_enabled or pad_pair_active
                ),
""",
        "Mamba intermediate pair states",
    )
    mamba_source = replace_once(
        mamba_source,
        """            else:
                # update ssm states
                # - varlen state is a (num_prefills, nheads, headdim, dstate)
                #   tensor
                ssm_state[state_indices_tensor_p] = varlen_states

        # Process decode requests
""",
        f"""            else:
                # update ssm states
                # - varlen state is a (num_prefills, nheads, headdim, dstate)
                #   tensor, except {MAMBA_MARKER} pair mode where it contains
                #   one entry for each chunk.
                if pad_pair_active:
                    first_state = varlen_states[0]
                    if (
                        self.voicechat_pad_pair_shadow_ssm is None
                        or self.voicechat_pad_pair_shadow_ssm.shape
                        != first_state.shape
                    ):
                        self.voicechat_pad_pair_shadow_ssm = torch.empty_like(
                            first_state
                        )
                    self.voicechat_pad_pair_shadow_ssm.copy_(first_state)
                    ssm_state[state_indices_tensor_p] = varlen_states[
                        last_chunk_indices_p
                    ]
                else:
                    ssm_state[state_indices_tensor_p] = varlen_states

        # Process decode requests
""",
        "Mamba pair-state commit and shadow",
    )
    mamba_source = replace_once(
        mamba_source,
        """    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
""",
        f"""    def voicechat_restore_pad_pair_state(self) -> bool:
        \"\"\"Restore the exact first-position state after PAD rejection.\"\"\"
        # {MAMBA_MARKER}: called only by the config-gated rejection branch.
        if (
            self.voicechat_pad_pair_shadow_conv is None
            or self.voicechat_pad_pair_shadow_ssm is None
            or self.voicechat_pad_pair_shadow_index is None
        ):
            return False
        self_kv_cache = self.kv_cache[
            self.voicechat_pad_pair_shadow_virtual_engine
        ]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        shadow_index = self.voicechat_pad_pair_shadow_index
        conv_state[shadow_index] = self.voicechat_pad_pair_shadow_conv.unsqueeze(0)
        ssm_state[shadow_index] = self.voicechat_pad_pair_shadow_ssm.unsqueeze(0)
        return True

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
""",
        "Mamba rejected-pair restore method",
    )
    mamba_path.write_text(mamba_source, encoding="utf-8")
    py_compile.compile(str(mamba_path), doraise=True)
    mamba_after = hashlib.sha256(mamba_source.encode()).hexdigest()
    print(f"Patched: {mamba_path} before={mamba_before} after={mamba_after}")


if __name__ == "__main__":
    main()
