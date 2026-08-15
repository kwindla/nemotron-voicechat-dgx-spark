# Adversarial review — model and runtime overview

Scope: uncommitted `docs/model-and-runtime-overview.md`,
`docs/model-and-runtime-overview.html`, and the two uncommitted `README.md`
hunks. I treated the dirty Speech checkout as untrusted and used
`git -C /home/khkramer/src/Speech-nemotron-voicechat show HEAD:<path>` for
all stock-code citations. Line numbers on those citations are from piping that
immutable blob through `nl -ba`.

## Summary

| Verdict | Count |
|---|---:|
| CONFIRMED-WRONG | 7 |
| UNSUPPORTED | 2 |
| MISLEADING | 10 |
| OMISSION | 2 |
| NITPICK | 4 |
| **Total** | **25** |

## Findings

### 1. Introduction — “the first open full-duplex model that can call tools” → MISLEADING

**Claim:** “is ... the first open full-duplex model that can call tools”
(`docs/model-and-runtime-overview.md:15-19`).

**Evidence:** NVIDIA makes this claim in the current model card
(`/home/khkramer/models/NVIDIA-NemotronLabs-VoiceChat-11B/README.md:23-28`)
and in Speech (`git show HEAD:README.md:10-14`), but neither source is an
independent comparison proving priority. The overview changes NVIDIA's
self-attribution into an unqualified fact.

**Exact correction:** “NVIDIA describes VoiceChat as the first open
full-duplex model to support tool calling.” Apply the same qualification to
`README.md:6-7`, which repeats the unqualified claim.

### 2. Introduction/model description — fixed-voice limitation omitted → OMISSION

**Claim/context:** The component table mentions a 3-second “Aria” speaker
prompt (`docs/model-and-runtime-overview.md:59`), but the document never tells
the reader whether that voice is configurable.

**Evidence:** The stock Speech README states that the released checkpoint has
one fixed voice and does not support voice cloning (`git show HEAD:README.md:16`).
That materially changes how a reader should interpret “speaker-prompt latent.”

**Exact correction:** Add: “The released checkpoint has one fixed voice and
does not support voice cloning; the Aria prompt is checkpoint/runtime state,
not a user-selectable cloning interface.”

### 3. Model consequences — “session state is dominated by recurrent SSM state” → UNSUPPORTED

**Claim:** “the session state is dominated by recurrent SSM state rather than a
KV cache” (`docs/model-and-runtime-overview.md:110-114`).

**Evidence:** Nano does have 52 Mamba layers and four attention layers, as
shown by `/home/khkramer/models/NVIDIA-Nemotron-Nano-9B-v2/config.json`, but
layer count does not establish which state consumes more bytes at the served
context length. No state-memory measurement or calculation is cited. KV state
also grows with context while recurrent state does not.

**Exact correction:** “Nano maintains substantial recurrent Mamba state in
addition to KV state, so this serving implementation preserves a live stateful
request rather than replaying stateless chat completions.” Do not claim
memory/state dominance without a byte calculation at a named context length.

### 4. Stock batch assumptions — “request migration impossible anyway” → MISLEADING

**Claim:** “52 Mamba layers of per-stream recurrent state make request
migration impossible anyway” (`docs/model-and-runtime-overview.md:249-254`).

**Evidence:** The Triton package is indeed batch-disabled and stateful
(`/home/khkramer/models/nemotron-voicechat_v26.01.20.1/nemotron-voicechat/config.pbtxt:3,29-66`),
but recurrent state makes migration unsupported by this implementation, not
impossible in principle; state can be serialized or copied by a system designed
to do so.

**Exact correction:** “This implementation does not support request migration:
Triton pins sequence state to one batch-1 instance, and neither the wrapper nor
the serving protocol serializes the Mamba/KV state needed to migrate it.”

### 5. Turn-taking — “special tokens bypass sampling entirely” at temperature zero → MISLEADING

**Claim:** “Sampling is greedy (temperature 0), and special tokens bypass
sampling entirely” (`docs/model-and-runtime-overview.md:156-159`).

**Evidence:** At temperature zero the stock sampler returns `argmax` for every
token before consulting `special_token_ids` (`git show
HEAD:nemo/collections/speechlm2/inference/model_wrappers/model_factory.py:104-120`;
the native implementation does the same at
`git show HEAD:nemo/collections/speechlm2/models/duplex_stt_model.py:125-140`).
The special-token branch matters only when non-greedy sampling is active.

**Exact correction:** “Production uses temperature-zero argmax for all tokens.
If non-greedy sampling is enabled, a raw special-token argmax bypasses penalties
and stochastic sampling.”

### 6. Tool calling — “one token per 80 ms frame” conflates positions with pacing → MISLEADING

**Claim:** The pipeline “force-injects the result one token per 80 ms frame,”
and a 73-position result therefore occupies “~6 s of real frames”
(`docs/model-and-runtime-overview.md:196-208`).

**Evidence:** NVIDIA's stock configuration explicitly calls FC async mode “full
LLM speed” (`git show
HEAD:examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml:115-121`).
The local benchmark found about 80.64 ms per injected *model position* on GB10
(`docs/tool-call-latency-fix-plan.md:51-60`), but the async loop is not paced by
an 80 ms wall clock. The measured similarity is a performance result, not the
semantic frame clock.

**Exact correction:** “The result consumes one model position per token.
Async FC advances those positions unpaced; on GB10 it measured about 80.64 ms
per position, so 73 positions took about 6.24 seconds.”

### 7. Stock loop diagram — silence substitution is shown on the wrong data path → CONFIRMED-WRONG

**Claim:** The diagram orders `EarTTS → Silence substitution on EOS/idle PAD →
Codec decode` (`docs/model-and-runtime-overview.md:219-229`).

**Evidence:** Stock code appends `code.clone()` to `new_codes_for_decode`
*before* substitution (`git show
HEAD:nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py:2546-2560`).
EOS/PAD substitution then changes the recurrent `code` fed to the next EarTTS
position (`ibid.:2561-2603`); codec decode consumes the earlier
`new_codes_for_decode` (`ibid.:2644-2677`). Thus the diagram falsely implies
that stock substitution changes the current decoded output frame.

**Exact correction:** Fork the EarTTS output: one cloned branch goes directly
to codec decode; the other passes through EOS/idle-PAD substitution and feeds
the next EarTTS recurrent input. Describe the later local idle-PAD decoder
bypass separately.

### 8. Stock loop — “three explicit cuda.synchronize barriers” → CONFIRMED-WRONG

**Claim:** “Three sequential model forwards, three explicit
`cuda.synchronize()` barriers, and a codec decode”
(`docs/model-and-runtime-overview.md:231-232`).

**Evidence:** Stock code synchronizes after perception (`git show .../nemotron_voicechat_inference_wrapper.py:2315-2337`),
Nano (`ibid.:2417-2448`), EarTTS (`ibid.:2532-2552`), **and codec decode**
(`ibid.:2644-2677`): four explicit barriers.

**Exact correction:** “Three sequential model forwards, four explicit CUDA
synchronizations (including one after codec decode), and a codec decode.” Add
`+ cuda.synchronize` to the codec node in the Mermaid diagram.

### 9. Deployment sequence diagram — wire event names are abbreviated as if exact → NITPICK

**Claim:** Pipecat sends `turn_start` and `commit`
(`docs/model-and-runtime-overview.md:347-356`).

**Evidence:** The actual strict-v3 wire messages are
`input_audio_buffer.turn_start` and `input_audio_buffer.commit`
(`src/nemotron_voicechat_pipecat/events.py:86-104`; see also
`docs/protocol-v3.md`). The prose immediately above calls them barriers but
does not say the diagram abbreviates their names.

**Exact correction:** Use the exact names in the diagram, or label them
“`input_audio_buffer.turn_start` (abbrev. turn_start)” and similarly for commit.

### 10. Browser sequence diagram — Smart Turn duration is okay, but the timing label is ambiguous → NITPICK

**Claim:** “Smart Turn v3 decides the end (~30 ms)”
(`docs/model-and-runtime-overview.md:354`; glossary at :732).

**Evidence:** Retained microphone measurements put Smart Turn v3.2 at about
28–35 ms (`docs/voice-to-voice-latency.md:18-23`). The much larger ~378 ms
browser endpoint line elsewhere includes VAD, Smart Turn, RTVI, and callback
delivery (`docs/model-and-runtime-overview.md:662`).

**Exact correction:** Say “Smart Turn classifier inference ~28–35 ms in the
retained run; the complete endpoint/VAD/RTVI interval is much larger.” This
prevents readers from treating 30 ms as end-to-end endpointing.

### 11. Reproduction diagram — “Signed replay corpus” lacks a named signature boundary → MISLEADING

**Claim:** “Signed replay corpus” (`docs/model-and-runtime-overview.md:432-439`).

**Evidence:** The corpus is comprehensively hash-bound by
`config/calibration-corpus-production-candidate-1.json`, and
`docs/weight-reproduction.md:20-35` calls its provenance manifest signed. The
diagram, however, makes the replay payload itself sound independently
cryptographically signed without naming the signer, signature artifact, or
verification step. The checked-in evidence directly demonstrates hashes and a
signed release/manifest chain.

**Exact correction:** “Hash-inventoried replay corpus, bound by the signed
release/corpus manifest (3,934 + 1,172 calls).” Link or name the manifest whose
signature is verified.

### 12. Serving loop — “12 buffered → 4 replayed, 290 ms saved” → CONFIRMED-WRONG

**Claim:** The onset selector replayed 12 → 4 frames, “290 ms saved”
(`docs/model-and-runtime-overview.md:459-464`).

**Evidence:** The retained evidence says 290.485 ms was the wall-time bracket
for replaying the **four retained serial positions**, not the saving
(`docs/voice-to-voice-latency.md:108-115`). The observed first-turn improvement
was 2260.8 → 1461.3 ms, or 799.5 ms, and the source correctly avoids assigning
all of that delta to the selector.

**Exact correction:** “12 buffered → 4 replayed; those four replay positions
took 290.485 ms. In the combined eager-start/onset run, first-turn V2V improved
by 799.5 ms; that cross-run delta is not an isolated selector saving.”

### 13. CPU codec attribution — “thousands of scalar grouped-convolution calls” → UNSUPPORTED

**Claim:** ARM64 PyTorch “fall[s] back to thousands of scalar
grouped-convolution calls” (`docs/model-and-runtime-overview.md:489-490`).

**Evidence:** The implementation does verify exactly 18 depthwise and 38
pointwise replacements (`src/nemotron_voicechat_runtime/cpu_codec_worker.py:70-90`),
and the retained latency work supports the offload. I did not find a retained
profiler trace or source-level calculation establishing “thousands” of scalar
calls. Grouped/depthwise fallback being slow does not itself prove that exact
mechanism or magnitude.

**Exact correction:** Either cite a retained profiler/call-count artifact, or
say “the ARM64 grouped/depthwise Conv1d path was pathologically slow; the worker
replaces exactly 18 depthwise and 38 pointwise convolutions with qualified
exact-math paths.”

### 14. Prepared epoch — “nine identity conditions” omits enforced identities → MISLEADING

**Claim:** A BOS may consume the epoch only if “nine identity conditions” match,
listing request, iterator, token count, speaker inputs, prompt IDs, initial
code, session epoch, prepare epoch, and turn
(`docs/model-and-runtime-overview.md:521-530`; glossary at :730).

**Evidence:** `_request_snapshot()` also binds a generation-unique physical
`backend_request_id`, `backend_generation`, and `request_state_id`, alongside
the iterator/token/speaker/prompt/code values
(`src/nemotron_voicechat_runtime/runtime_optimizations.py:75-110`). `_matches()`
additionally checks `session_epoch` and `prepare_epoch` (`ibid.:115-136`), and
arming binds the client turn (`ibid.:261-300`). The prose's arbitrary count
hides the ABA-safety identities added specifically to prevent stale EarTTS
requests.

**Exact correction:** Remove the count and list every bound category, including
logical request, physical backend request and generation, RequestState object,
iterator, generated-token count, speaker/prompt/init-code fingerprints,
session/prepare epochs, and committed turn.

### 15. Cold startup attribution — the supposedly additive abbreviated list does not add to 382.9 s → CONFIRMED-WRONG

**Claim:** “Cold start was attributed additively: Nano 154 s, EarTTS 119 s,
NeMo 41 s, pipeline setup ~47 s, warmup 10 s — 382.9 s”
(`docs/model-and-runtime-overview.md:535-539`).

**Evidence:** Those rounded terms total about 371 s, not 382.9 s. The actual
additive account contains additional hydration, inter-engine setup, exact
prompt warmup, application/Uvicorn startup, and health-poll phases
(`reports/step9-cold-start-20260807.md:46-78`). “Pipeline setup ~47 s” only
combines pre-model 33.968 and post-engine 13.059 seconds and does not absorb all
omitted phases.

**Exact correction:** Call the abbreviated list “largest contributors,” or
reproduce the complete additive table whose sum is 382.882994 seconds.

### 16. Tool-call fast grace — current p95 is not 0.957 ms → CONFIRMED-WRONG

**Claim:** “result-available → result injection p95 fell ... to **0.957 ms**”
as one of “Two fixes shipped on the current branch”
(`docs/model-and-runtime-overview.md:615-622`).

**Evidence:** 0.957 ms is the step-one candidate result
(`docs/tool-call-latency-fix-plan.md:129-143`). The final image measured 10.30
ms and explicitly preserved the difference as image/load variance
(`docs/tool-call-latency-fix-plan.md:244-262`). Both pass the 250 ms gate, but
the overview substitutes the better predecessor number for current evidence.

**Exact correction:** “Step one measured 0.957 ms p95; the final step-two image
measured 10.30 ms p95. Both are far inside the unchanged 250 ms gate.”

### 17. Post-fix browser smoke — not a tool-call latency measurement → MISLEADING

**Claim:** Under “Tool-call latency,” “Post-fix browser smoke: 1.387 s warm
voice-to-voice with a tool configured” (`docs/model-and-runtime-overview.md:631-632`).

**Evidence:** The retained report describes a browser smoke that exercised a
real clock tool plus voice/typed input, but the 1.387-second point is a warm
voice-turn latency sample in a tool-advertised session, not the 72-token
SOTC-to-post-tool interval (`docs/tool-call-latency-fix-plan.md:269-278`).

**Exact correction:** “A non-tool warm voice turn in the browser smoke, in a
session that advertised a tool, measured 1.387 s V2V; this is not end-to-end
tool-call latency.” Report the tool-cycle interval separately if available.

### 18. Chained tools — “upstream checkpoint limitation” is not established → CONFIRMED-WRONG

**Claim:** Chained calls degrade by call three and are “an upstream checkpoint
limitation” (`docs/model-and-runtime-overview.md:634-638`).

**Evidence:** The retained plan says the one-call chained limitation reproduced,
but explicitly says the project still has to distinguish a prompt/turn-history
defect from the model limitation (`docs/tool-call-latency-fix-plan.md:77-83`).
The final result only establishes 0/3/non-promotable behavior
(`docs/tool-call-latency-fix-plan.md:269-272`), not causality.

**Exact correction:** “Chained calls remain unreliable and non-promotable;
degradation is visible by call three, but whether the cause is prompt/history
handling or checkpoint capability remains unresolved.”

### 19. Latency milestone table — it visually implies an additive optimization series → MISLEADING

**Claim:** Rows prefixed with “+ eager start,” “+ idle-PAD,” and “+ prepared
epoch” form the “Current measured latency budget”
(`docs/model-and-runtime-overview.md:640-652`).

**Evidence:** These are point samples from different immutable images and runs,
not controlled deltas or a distribution. The source explicitly says the warm
128.7 ms difference is run variance (`docs/voice-to-voice-latency.md:176-181`)
and that the final ngrok point cannot isolate the approximately 75 ms prepared
reset saving (`docs/voice-to-voice-latency.md:313-326`).

**Exact correction:** Remove the “+” ladder notation. Label every row with its
image/run and “point sample,” and add: “Differences between rows are not
isolated causal savings and are not percentile/SLA evidence.”

### 20. Latency conclusion — “architectural, not tunable” overstates a qualified current contract → MISLEADING

**Claim:** “The residual structure is architectural, not tunable”
(`docs/model-and-runtime-overview.md:673-679`).

**Evidence:** The ten-blank fence is a currently qualified correctness
contract, and the source discusses alternative boundary designs, overlap with
endpointing, and further profiling (`docs/voice-to-voice-latency.md:236-282`).
Those require redesign and requalification, but are not physically immutable.

**Exact correction:** “Under the current qualified boundary and output
contracts, the residual cannot be removed by parameter tuning alone; reducing
it requires a redesigned boundary/onset path and full requalification.”

### 21. Qualification — “every promoted claim ... every number ... one immutable runtime image” → MISLEADING

**Claim:** “Every promoted claim is bound to an immutable runtime image and
retained traces; every number in this document traces to one”
(`docs/model-and-runtime-overview.md:698-700`).

**Evidence:** Architecture numbers come from checkpoint/config metadata, stock
mechanism claims come from immutable Git blobs, and some latency numbers are
historical point samples across multiple images. A single runtime-image rule is
neither applicable to all claims nor true of the document as phrased.

**Exact correction:** “Every promoted runtime-performance claim is bound to a
named immutable image and retained evidence; architecture and stock-code claims
are bound to the frozen checkpoint/config and Git revisions identified below.”

### 22. Frozen identities — abbreviated vLLM hashes are weaker than the surrounding table → NITPICK

**Claim:** Native/fork vLLM identities are `b31e9326` / `237ba3ca`
(`docs/model-and-runtime-overview.md:754-762`).

**Evidence:** The exact revisions are
`b31e9326a7d9394aab8c767f8ebe225c65594b60` and
`237ba3cafc14d514551539ad980f17d40d273dfa`
(`config/qualified-candidate-1.json`, `public_sources.native_vllm.revision` and
`public_sources.voicechat_vllm.revision`). The table otherwise uses full
checkpoint/Speech revisions.

**Exact correction:** Print the full 40-character vLLM revisions too.

### 23. HTML wrapper — claimed repeated-render idempotence has an overlapping-call race → CONFIRMED-WRONG

**Claim:** `renderMermaid()` runs on every `md-render` and “is idempotent”
(`docs/model-and-runtime-overview.html:66-90`).

**Evidence:** The function is async and has no in-flight lock or rendered
marker. The unconditional call at line 90 can overlap an `md-render` callback
at line 89. Both invocations can collect the same `<code>` nodes at lines
72-76; after one awaits `mermaid.render()` and replaces the `<pre>`, the other
reaches `code.closest("pre").replaceWith(host)` at line 85 with `closest()` now
null, causing an uncaught `TypeError`. The `try` only covers rendering, not the
replacement.

**Exact correction:** Serialize renders (one promise/queue), mark or remove a
fence before the first await, and null-check the parent before replacement.
Alternatively remove the unconditional call and rely on one well-defined
post-fetch event after verifying md-block's lifecycle.

### 24. HTML/README — direct opening and load failures are not handled → OMISSION

**Claim:** README says to “open `docs/model-and-runtime-overview.html` in a
browser” (`README.md:153-157`); the HTML presents “Loading document…” and loads
the Markdown plus two modules (`docs/model-and-runtime-overview.html:55-58`).

**Evidence:** `<md-block src="model-and-runtime-overview.md">` requires fetch,
which commonly fails under `file://`; both md-block and Mermaid are remote CDN
dependencies. There is no fetch/module error handler, timeout, or status update,
so offline/CDN/CORS failures can leave “Loading document…” forever. The Markdown
itself correctly says to serve the directory (`docs/model-and-runtime-overview.md:9-11`),
but README omits that operational requirement.

**Exact correction:** README should say: “Serve `docs/` over HTTP (for example,
`cd docs && python3 -m http.server`) and visit the resulting
`model-and-runtime-overview.html` URL.” Add visible failure handling for md-block
load and module/render errors, and disclose that the wrapper requires network
access to its two CDN modules (or vendor them for offline use).

### 25. HTML wrapper — Mermaid theme is fixed at module-load time → NITPICK

**Claim/mechanism:** CSS follows `prefers-color-scheme`, while Mermaid is
initialized from one `matchMedia(...).matches` snapshot
(`docs/model-and-runtime-overview.html:17-25,60-61`).

**Evidence:** There is no `matchMedia` change listener and no diagram rerender.
If the OS/browser theme changes while the page is open, the surrounding page
switches themes but the already-generated SVGs retain the old Mermaid theme.
Initial light/dark selection is correct; live theme consistency is not.

**Exact correction:** Either state that diagram theming is fixed until reload,
or listen for color-scheme changes, reinitialize Mermaid, and rerender from
retained source text through the same serialized render path required by
finding 23.

## Spot-checks that passed

The following claims were checked against primary sources and did **not**
produce findings:

- **Checkpoint size, dtype, and exact parameter accounting.** Parsed the
  44,382,749,892-byte safetensors header: 1,632 tensors and 11,095,371,286
  elements. F32 parameters total 11,095,109,105; the remaining 262,181 I64
  elements are buffers. Prefix sums reproduce the document's rounded 7,714 M
  Nano, 797 M EarTTS, 614 M perception, three 587 M matrices, 200 M codec, and
  9 M RNNT totals.
- **Nano architecture.** The frozen Nano config confirms 56 layers, 52 Mamba2
  plus four attention layers, hidden 4480, MLP 15680/relu², 40 query and eight
  KV heads of dimension 128, and vocabulary 131,072.
- **EarTTS architecture.** The public checkpoint config confirms 28 layers,
  hidden 1152, intermediate 4608, 16/16 heads with dimension 72, three-layer
  low-rank-64/1024 MoG head, guidance 0.2, 31 quantizers, 3-second Aria prompt,
  and one T5Gemma encoder layer. The served converted config confirms window
  1500 and full attention at layers 5, 11, 17, and 23.
- **Perception/RNNT architecture.** The combined config confirms 128 mels,
  25/10 ms window/stride, 24 × 1024 FastConformer, depthwise ×8 subsampling,
  context `[70,0]`, 640-d two-layer RNNT prediction network, and the 1024-token
  vocabulary. The 10 ms × 8 = 80 ms arithmetic passed.
- **Codec arithmetic.** 22,050 / 1,764 = 12.5 frames/s; 31 codebooks × 10 bits ×
  12.5 = 3,875 bit/s. The 31 × 1024-code/latent-512 config values passed.
- **Four-channel fusion and heads.** The combined config has weights 1/1/1/2
  and `predict_user_text=false`; stock `duplex_stt_model.py` constructs
  `function_head = copy.deepcopy(lm_head)`. RNNT rewriting occurs after Nano
  and before EarTTS in the immutable wrapper.
- **NVIDIA attribution.** The current model card verifies release date
  2026-08-03, OpenMDW-1.1, 11B/~450 ms/#2 marketing fields, and the listed
  A100/H100/H200/B100/B200/RTX-6000 hardware. Stock Speech separately states an
  80 GB GPU minimum (`git show HEAD:README.md:18-20`). The stock YAML explicitly
  says FP32 EarTTS is required because BF16 causes hallucinations
  (`git show HEAD:.../s2s_streaming.yaml:42-50`).
- **Stock turn-taking defaults.** The stock YAML confirms RNNT EOU/BOU 40 and
  tool timeout 15 seconds; the immutable vLLM inference script defaults its
  `S2S_RNNT_EOU_FRAMES` override to 15; production config uses the separately
  documented legacy 20-frame mode. The overview correctly distinguishes that
  from the qualified 10-blank settlement fence.
- **Stock vLLM request settings.** Verified temperature 0, `ignore_eos=True`,
  `max_tokens=100000`, FP32 Mamba cache, Nano/EarTTS memory fractions 0.52/0.18,
  and batch-disabled Triton sequence state.
- **Protocol bounds and capabilities.** `protocol.py` confirms 16 KiB tool
  output, 128 model-output tokens, 256 recovery frames, eight calls per
  response, and `client_smart_turn_v1`; production advertises client-owned
  endpointing. The server executor is exactly one worker
  (`server.py:5806-5808`).
- **Typed input and transport layout.** Confirmed Pocket runs process-isolated
  and CPU-only, typed PCM traverses perception/RNNT, codec worker cores are 5/6,
  Pocket cores are 7/8/9/15, the model server is loopback port 8786, Pipecat is
  port 7860, and Pipecat's default output prebuffer is 160 ms
  (`src/nemotron_voicechat_pipecat/llm.py:130,168-178,1230-1242`).
- **Quantization recipe and gates.** Verified GPTQ W8/group 128/damp 0.01,
  exactly 105 Nano tensors, attention/function/embedding exclusions,
  15680→15744 g128 padding, exactly 205 EarTTS tensors, 3,934/1,172 corpus
  calls, and gate values 1.0/0.998677/1.0/0.3125. The W8A32 implementation
  promotes/accumulates FP32 and registers a `torch.library` op.
- **Reproduction identities.** Source revision `fb0f94e...`, Spark revision
  `a20c685...`, Speech `911ec674...`, release hash `1d8c3d...`, and the two
  byte-identical/hash-identical conversion requirement match config and
  `docs/weight-reproduction.md`.
- **CPU codec mechanics.** Exact replacement gates of 18 depthwise and 38
  pointwise convs, worker prewarm, CPU-only subprocess, core pinning, and
  one-frame pipeline behavior passed.
- **Startup final numbers.** The full report verifies 382.882994 seconds to
  247.954/254.087 seconds, savings of 33.6–35.2%, and the rejected ~260-second
  exact-prompt result.
- **PAD-pair mechanism, numbers, and retirement.** The 1,389/1,389 bitwise gate,
  68.24/71.13 mean, 84.77/73.98 p95, 94.79/84.64 p99, 510-frame degraded
  interval, 2.98-second debt, retirement commit `d4a7e0a`, and production policy
  rejection of all pair flags matched the retained docs, patches, and history.
- **Tool-call benchmark arithmetic.** Verified ~11.2 seconds total, 2.32 call
  emission, 2.08 on-hold, 6.24 result injection, 0.56 onset, the
  `355.9 + 80.64×positions` regression with R² 0.9992, ~60.7 ms Nano execution,
  21-position concise output, and 1.760-second mean recovery.
- **Latency measurements.** Reconciled the six-position totals
  70.708/186.495/96.836/45.012/12.008 = 411.059 ms, the 1470.7/732.341 baseline,
  1360.0/1322.5 bypass samples, and 1343.5/552.2/795.7 ngrok sample. The Gantt
  sums exactly 378+48+430+181+248+123 = 1,408 ms and correctly rounds the
  source's 1,407.5-ms turn.
- **Idle-PAD/prepared-epoch mechanics.** Verified 439 bypass positions at zero
  EarTTS time, recurrent identity preservation, 72.9–78.3 ms BOS resets,
  single-use arm consumption, fallback behavior, and the accepted exclusive
  transition tuples `(1,1,0)` and `(1,0,1)`.
- **Qualification tree count.** `config/qualified-candidate-1.json` classifies
  838 VoiceChat-equal + 84 equal-to-both = 922 identical, 36 deltas, and eight
  additions, totaling 966 exactly as documented.
- **Mechanism/file cross-references.** Every basename in §7.2 exists in the
  stated runtime/conversion area; the retired paths remain present for offline
  diagnosis. The README's first hunk accurately replaces the obsolete claim
  that conditional PAD drafting is the current production strategy.
