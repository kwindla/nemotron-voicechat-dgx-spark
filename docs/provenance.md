# Production Candidate 1 provenance

## Public model

- Repository: `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`
- Revision: `fb0f94eaf4d03ddc430f39565229393fa1b50c26`
- Combined model SHA-256:
  `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`
- Derived Nano: calibrated symmetric GPTQ W8, group 128; 105 quantized
  MLP/Mamba/text-head tensors; function head and embeddings full precision.
- Derived EarTTS: symmetric per-output-channel W8 weights with FP32
  activations/accumulation; 1,500-position configured sliding window.
- Both derived conversions were byte-identical across two runs in the retained
  source manifest. No weights derive from the retired EA checkpoint.
- Redistribution target:
  `pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`. OpenMDW-1.1 permits
  redistribution of the derived model materials when its license and origin
  notices are retained. This is labeled as a community quantization, not an
  NVIDIA release.
- The exact fused-embedding calibration corpus is redistributable: its prompts
  are project-authored, its tensors are model outputs, and it contains no raw
  user recording or personal data. Its content-addressed identity is
  `ad8ce2fe3a66fae72ce45703216265180066b2745dc32d829ee550b45c487bd5`.

The serialized EarTTS quantization method retains the historical internal name
`ea_w8a32`. That string is an artifact ABI, not support for the EA checkpoint.
Renaming it would require regenerating and requalifying the derived model.

## NVIDIA Speech

- Repository: `https://github.com/NVIDIA-NeMo/Speech.git`
- Commit: `911ec674ab40f04302ef33672be4179f45a7310f`
- Retained patch:
  `src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch`
- Runtime-opened template:
  `examples/speechlm2/function_calling/template.jinja`

The server refuses startup unless the three reviewed Speech files have exactly
the retained diff.

## vLLM and runtime image

- NVIDIA public Voicechat fork reference:
  `vklimkov-nvidia/vllm@237ba3cafc14d514551539ad980f17d40d273dfa`
- Historical comparison-oracle image tag:
  `nemotron-local/voicechat-vllm:public-candidate1-hotfix5-20260803`
- Historical comparison-oracle image ID:
  `sha256:fa11b6ef4ea2a8d6c1cc89f7c26ea5af2b98bb9421dd9867caa8ae95227b5c09`
- Qualified public-build image tag:
  `pipecat-ai/nemotron-voicechat-dgx-spark:production-candidate-1`
- Qualified public-build image ID:
  `sha256:6294053537329500a17a1684421cd3909224babd7423d677c6c5d6bceead0482`
- Qualified public-build SBOM SHA-256:
  `a065741de28dfddb10c57c9fc30fd206c4edb2931c8fc7e2f4691c93cd6197b2`
- Driver: 580.142
- Kernel: 6.17.0-1014-nvidia

The historical image is comparison evidence only. The qualified release image
was rebuilt with `--no-cache` from the pinned public NVIDIA PyTorch base and
public source graph. Its complete history contains no historical EA/NGC layer,
and the no-cache build reproduced the cached build's SBOM byte-for-byte.

The calibration replay metadata preserves the capture-time label
`nemotron-local/voicechat-vllm:public-runtime`. It is an informational record,
not a resolvable input. Conversion consumes only the content-addressed replay
tensors and the public build graph recorded here.

## Pipecat

- Reference and dependency commit:
  `5415f28af9407937610d68abb2c195365650a862`
- The implementation follows Pipecat's OpenAI Realtime and Gemini Live service
  lifecycle patterns without modifying Pipecat core.

## Pocket TTS typed input

- Package: `pocket-tts==2.1.0`, installed from the in-image Pocket constraints
  lock.
- Runtime: `torch==2.9.1+cpu` from the official PyTorch CPU wheel index; no
  CUDA libraries are part of the isolated `/opt/pocket-tts` environment. Host
  Pipecat does not install Pocket TTS or PyTorch.
- Repository: `kyutai/pocket-tts-without-voice-cloning`.
- English 2026-04 model/tokenizer revision:
  `d29db7978e464fb90cb3359ee0c69a273b9142cc`.
- `model.safetensors`: 219,029,196 bytes, SHA-256
  `be9c6b4876d3f30740a8225dfcaa2e43dc4aeb753c15272735bee16bbb4abb0a`.
- `tokenizer.model`: 59,339 bytes, SHA-256
  `d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6`.
- Precomputed `alba` embedding revision:
  `e041936c75475d350b405bc870bcf7c22da4e9e6`.
- `alba.safetensors`: 6,194,424 bytes, SHA-256
  `69c32db63ca56843d994f81f343f62e0bf2d73f7e4c9bc73e44bb1110b1d8845`.

Production runs with Hugging Face offline mode. These public artifacts must be
present in the verified host cache before the foreground model container starts;
the cache is mounted read-only. The bridge uses Pocket TTS's supported
dynamic-int8 path, four CPU threads, and a cached voice state. Release
qualification re-measures prewarm and real-time factor inside the container;
the former host-side 1.95x result is retained only as retired-architecture
evidence.

## Initial extraction source

- Private source worktree branch at extraction start:
  `codex/pipecat-voicechat-service`
- Private source commit at extraction start:
  `addb8b2` (`docs: record Pipecat qualification phase status`)
- Fable reviewed the extraction plan before implementation. The full review
  history is retained in the private development repository; this public tree
  carries the resulting signed provenance and operational evidence. These
  names describe the initial extraction only and are not runtime inputs.
