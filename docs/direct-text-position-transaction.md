# Direct user-source position transaction

Status: Step 2A implementation inventory. This document describes the exact
mutable owners in the pinned NVIDIA Speech runtime and is normative for the
diagnostic transaction seam. A field not listed here must not be mutated by a
direct position until this inventory and its assertions are updated.

## Transaction boundary

`StreamingS2SPipeline.inject_user_source_position(stream_id, source_token_id)`
is the only model-level entry point. It represents one 80 ms VoiceChat timeline
position whose source-channel value is a tokenizer embedding instead of a
perception embedding. The call is serialized by the server's existing model
lock. It performs all validation before owner mutation number one. Once Nano is
called, failure is not recoverable because vLLM continuation state has no
rollback; the server must terminate the session.

The Step 2A/2B prototype rejects the call unless all of these conditions hold:

- an already-prefilled, active batch-one stream and slot context exist;
- `num_frames_per_chunk == 1`, the token ID is in vocabulary, and the context
  has room for this position plus the later explicit-EOU position;
- the agent and function-call state are idle, no FC background worker exists,
  and no external user-EOU, agent-EOS, redirect, or post-tool BOS latch is set;
- Nano and EarTTS vLLM requests both exist and are active;
- `VOICECHAT_NANO_PAD_PAIR` is disabled and the stream has no buffered draft;
- the source tensor can be fused to BF16 shape `[1, 1, 4480]`.

Admission limits, an open microphone/typed input turn, and the response/function
epoch are server owners. `VoiceChatEngine` checks those conditions before it
calls the pipeline seam. The pipeline repeats every condition it can observe so
a future caller cannot create a partial transaction.

## Mutable-owner inventory

| Owner | Exact fields | Direct-position decision and assertion |
|---|---|---|
| `StreamingS2SPipeline` | `context_manager`, `_stream_has_prompt`, `_fc_async_bg`, `bufferer`, `_state_pool` | Reuse the existing context. `_stream_has_prompt`, `_fc_async_bg`, and `bufferer` are unchanged. The pre-existing `S2SStreamingState` in `_state_pool[stream_id]` is read but receives no display output or delivered audio. |
| `S2SContextManager` maps | `streamidx2slotidx`, `slotidx2streamidx`, `free_slots`, `slot_contexts` | All maps/queue membership are unchanged. The existing `slot_contexts[slot]` object is updated in place only after one wrapper result exists. |
| `StreamingRealtimeContext` timeline | `frame_idx`, `perception_frame_idx`, `gen_text`, `gen_asr_text`, `gen_function_text`, `subword_mask` | `frame_idx` advances by exactly one while `perception_frame_idx` does not. At the old timeline index all present generated channels are forced to `text_pad_id`; `subword_mask` is set `True`. Earlier and later cells are byte-identical. The next injected position therefore receives PAD feedback; the first receives the exact prior live agent/ASR/function tokens (or the normal position-zero embeddings). |
| Nano continuation | `dynamic_cache`, `input_embeds_history`; vLLM `model_llm_interface.engine.requests[request_id].generated_tokens`, `.generation_iterator`, `.status` | The qualified runtime uses vLLM, so `dynamic_cache` and `input_embeds_history` remain unchanged. Nano consumes exactly one fused embedding and its generated-token count increases by one. Iterator identity/status remain active. Failure after this call is session-fatal. Native cache/no-cache engines are rejected by the prototype. |
| Source/perception | `perception_frame_idx`; `perception_cache.cache_last_channel`, `.cache_last_time`, `.cache_last_channel_len`, any other `PerceptionCacheState` members; wrapper `_last_user_audio_emb`; manager CUDA-graph state | Perception is not called, `perception_frame_idx` does not advance, and the entire `perception_cache` object graph is unchanged. `_last_user_audio_emb` becomes the injected source embedding because it is the most recent source-channel value; FC is required idle and the next real audio position replaces it. Manager/CUDA-graph state is unchanged. The next real-audio call uses `perception_frame_idx`, not the later Nano timeline `frame_idx`, so first/subsequent chunk selection remains correct. A post-injection real-audio component test is still mandatory. |
| RNNT | `rnnt_partial_hypotheses`, including `y_sequence`, `blank_count`, `nonblank_consec`, `nonblank_total`, `speech_confirmed`, `agent_speaking`, `first_turn`, `user_first_turn`, `rolling_density`, `forced_bos`, `post_eos_fired`, `_punct_word_acc`, `_punct_bias_val`, `pred_out`, `pred_hidden`, `_turn_text_tokens`, `_agent_talking_frames`; wrapper `_rnnt_consecutive_speech_frames`, `_rnnt_last_speech_frame`, `_rnnt_prev_num_tokens`, `_rnnt_bos_cooldown_until`, `_rnnt_eos_cooldown_until` | No RNNT encoder/decoder/turn-taking function is called. Every hypothesis member and wrapper RNNT counter is identity/value unchanged. The later common explicit-EOU audio position, not this direct position, consumes the EOU latch and opens the assistant turn. |
| Fusion input | `fusion_module`; source token embedding; previous `gen_text`, `gen_asr_text`, `gen_function_text` | Parameters are unchanged. Source uses `stt_model.embed_tokens(source_token_id)` in the trained user/source-audio slot. Agent/user-text/function inputs use exact previous context at the first injected position; after forced PAD output, later positions use PAD. Assert dtype/device and fused `[1, 1, 4480]`. Step 2B decides plain versus BOS/text/EOS token form; it never changes system prefill. |
| Function calling | `context.fc_state`, `tool_response_text`, `tool_response_queue`; `_fc_async_bg`; wrapper `_post_tc_bos_exempt`, FC token IDs/caches | The call is admitted only with an idle `fc_state` (`active`, `awaiting_response`, `injecting_response`, and `forced_function_tokens` false/empty), no background entry, and no queued response injection. Nano's raw function prediction is discarded and effective output is PAD. `fc_state`, response text/queue, and all wrapper FC latches/caches are unchanged. |
| EarTTS recurrence | `context.code`, `past_key_values`; EarTTS vLLM request `.generated_tokens`, `.generation_iterator`, `.status`; wrapper `first_context_subword_id`, `generation_config`, `_tts_in_turn_content`, `_tts_in_turn_pads` | EarTTS consumes exactly one effective PAD subword with exact prior subword/code feedback. It advances once and returns its actual recurrent code; that code is retained exactly as the ordinary idle-PAD path retains it. Native `past_key_values` is rejected in the prototype. Request count advances by one and status/iterator remain active. The PAD counters advance as in the ordinary path; agent-idle stays true. |
| Codec/audio recurrence | `context.codec_cache`, `audio_toks_buffer`; `S2SStreamingState.audio_buffer`; wrapper capture sequence | The normal codec decode runs once so overlap/cache state remains aligned with EarTTS, but decoded samples are discarded and never passed to `log_output`. `codec_cache` advances once; `audio_toks_buffer` follows the configured fallback path. `state.audio_buffer` remains empty and capture is rejected for the prototype. A sequential idle-PAD equivalence test compares recurrence and the first delivered post-injection frame. |
| Display/turn state | `S2SStreamingState.output_text_str`, `output_text_tokens`, `output_asr_text_str`, `output_asr_text_tokens`, `output_function_text_str`, `output_words`, `_last_sent_asr_text`, `final_*`, `fc_timing`, `agent_idle`; wrapper `_agent_idle` | No output logger runs, so display strings/tokens/words and final snapshots are unchanged. Both locked per-stream `agent_idle` and wrapper fallback remain true. `fc_timing.frame_step_times` may append one diagnostic timing row only after the transaction succeeds; no response-start event may be added. |
| Wrapper turn/transport latches | `_external_user_eou_mode`, `_external_user_eou_requested`, `_external_agent_eos_requested`, `_agent_eos_just_fired`, `_redirect_tokens_queue`, `_post_tc_bos_exempt`, `_transport_input_activity_available`, `_transport_user_speech_seen`, `_transport_input_silent_frames`, `_tts_in_turn_content`, `_tts_in_turn_pads` | External mode remains enabled. Immediately before pipeline preflight, `VoiceChatEngine` consumes a stale `_agent_eos_just_fired` edge only when no FC worker exists, exactly as the ordinary PCM path does before its next model step. This permits a legal next typed turn to retain the preceding EOS as exact feedback without exposing that edge to a future worker. Every other pending edge/redirect must be clear; an EOS edge with active FC remains set and is rejected. Transport activity evidence and wrapper turn counters are unchanged except the ordinary idle-PAD TTS pad counter described above. |
| PAD-pair scheduler | Nano engine `_voicechat_pad_pair_states[request_id]`, especially `pending`, `needs_correction`, `assumed`, `sequential_mode`, `previous_effective_pad`; `_voicechat_pad_pair_call_context` | Optimization must be disabled and any existing stream state must have `pending is None` before entry. All scheduler state is unchanged. The prototype never drains or creates a draft at the direct boundary. |
| `VoiceChatEngine` | `frame_index`, `audio_frame_index`, `assistant_position`, `function_position`, `user_text`, `user_text_prefix`, `user_text_segment`, `last_rnnt_decoded_count`, `vllm_request_position_baseline`, `agent_silence_watchdog`, `response_boundary`, pad-pair trace/watchdog counters | Model `frame_index` advances with the context only after success; `audio_frame_index` does not, so the next PCM frame still carries `is_first=True` and initializes bufferer/perception/RNNT metadata. Text/function offsets, accumulated user text, RNNT count, baselines, watchdogs, response boundary, and trace counters do not observe hidden injection positions. Per-position assertions require Nano and EarTTS each to advance once and preserve their pre-existing epoch offset. That offset is zero before the first assistant BOS. At BOS, the qualified `VOICECHAT_EARTTS_RESET_ON_BOS` optimization deliberately starts a new EarTTS acoustic request epoch; its one-time reset/baseline and subsequent constant offset from Nano's continuous conversation epoch are asserted explicitly. |
| Strict-v3 server | model lock, input arbiter/turn protocol, typed job mutation flag, response/function epochs | All preconditions run while holding the model lock. `input_text.injection_started` is the point of no return immediately before the first position. A failure before it is recoverable; a failure from the first Nano call onward closes the session. No input transcript or response event is emitted by an individual hidden position. |

## Snapshot and failure contract

The diagnostic seam records scalar counts, identities, shapes, and cloned small
token/latch values before validation. It does not clone KV, perception, RNNT, or
codec tensors as a rollback mechanism. Precondition tests use the same snapshot
function and require exact equality after every rejected call. Successful
component tests capture one before/after record per position and require:

1. `frame_idx`, Nano generated-token count, EarTTS generated-token count, and
   codec decode step each increase by one, while `perception_frame_idx` does
   not;
2. effective generated text/ASR/function cells equal PAD and no display/audio
   output is appended;
3. perception and every RNNT member are unchanged;
4. Nano and EarTTS each advance once and preserve their epoch offset throughout
   every hidden-injection epoch. The offset is zero before the first assistant
   BOS. At the BOS edge, the qualified
   runtime performs exactly one EarTTS abort/re-prefill and begins the acoustic
   epoch at position one; subsequent checks compare clocks relative to that BOS
   baseline rather than requiring absolute equality with Nano;
5. the first source position uses exact preceding feedback; and
6. pad-pair state is disabled, empty, and unchanged.

The transaction is deliberately forward-only. There is no claim of atomic
rollback after a vLLM append. “Atomic” means complete preflight before mutation,
one serialized advancement across every required owner, and mandatory session
termination on any post-mutation exception.
