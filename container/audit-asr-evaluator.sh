#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: audit-asr-evaluator.sh IMAGE}
image_id=$(docker image inspect --format '{{.Id}}' "${image}")
case "${image_id}" in
  sha256:????????????????????????????????????????????????????????????????) ;;
  *) echo "invalid evaluator image ID: ${image_id}" >&2; exit 1 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
expected_base='nvcr.io/nvidia/pytorch@sha256:1dc787f5c6264fcc1c99809f99b84823e73ed4588d5a581b94290fc2a8fecff8'
test "$(docker image inspect --format '{{index .Config.Labels "ai.pipecat.voicechat.role"}}' "${image_id}")" = 'independent-asr-evaluator'
test "$(docker image inspect --format '{{index .Config.Labels "ai.pipecat.voicechat.base-image"}}' "${image_id}")" = "${expected_base}"

docker run --rm --network none "${image_id}" python3 -m pip check

# Qualification report recomputation must remain usable in this deliberately
# minimal image without importing the live WebSocket client stack.
docker run --rm --network none \
  -v "${repo_root}:/workspace/project:ro" \
  "${image_id}" \
  python3 -P -c \
  'import sys; from transcribe_sustained import tool_freshness_runtime_gate_passed; assert tool_freshness_runtime_gate_passed({"input_modality":"typed"}) is False; assert "websockets" not in sys.modules'

docker run --rm -i --gpus all --ipc host --network none \
  -e "VOICECHAT_ASR_EVALUATOR_IMAGE_ID=${image_id}" \
  -v "${repo_root}:/workspace/source:ro" \
  "${image_id}" \
  python3 -P - <<'PY'
import importlib.metadata
from pathlib import Path

import numpy as np
import soundfile as sf
from nemotron_voicechat_asr_evaluator import (
    DEFAULT_MODEL,
    NemotronEnglishAsr,
    load_manifest,
    sha256_file,
    verify_snapshot,
)

manifest = load_manifest()
verify_snapshot(DEFAULT_MODEL, manifest)
canary = manifest['speech_canary']
assert sha256_file(Path(canary['audio_path'])) == canary['audio_sha256']
assert manifest['model_id'] == 'nvidia/nemotron-speech-streaming-en-0.6b'
assert manifest['revision'] == 'ebe59e5a817142986528bbbee5dba8db7b38ed50'
assert importlib.metadata.version('transformers') == '5.14.1'
assert importlib.metadata.version('tokenizers') == '0.22.2'
assert importlib.metadata.version('huggingface-hub') == '1.27.0'
assert importlib.metadata.version('safetensors') == '0.8.0'
assert importlib.metadata.version('typer') == '0.21.0'
assert importlib.metadata.version('click') == '8.4.2'
assert importlib.metadata.version('hf-xet') == '1.5.2'

comparisons = {
    Path('/opt/voicechat-asr/model-manifest.json'): Path('/workspace/source/config/nemotron-asr-en-0.6b.json'),
    Path('/opt/voicechat-asr/requirements.txt'): Path('/workspace/source/container/requirements-asr-evaluator.txt'),
    Path('/opt/voicechat-asr/nemotron_voicechat_asr_evaluator/backend.py'): Path('/workspace/source/src/nemotron_voicechat_asr_evaluator/backend.py'),
    Path('/opt/voicechat-asr/nemotron_voicechat_asr_evaluator/__init__.py'): Path('/workspace/source/src/nemotron_voicechat_asr_evaluator/__init__.py'),
    Path('/licenses/LibriSpeech-canary-NOTICE.md'): Path('/workspace/source/licenses/LibriSpeech-canary-NOTICE.md'),
}
for embedded, checked_in in comparisons.items():
    assert sha256_file(embedded) == sha256_file(checked_in), (embedded, checked_in)

backend = NemotronEnglishAsr()
audio, rate = sf.read(manifest['speech_canary']['audio_path'], dtype='float32')
assert rate == 16_000 and audio.ndim == 1
first = backend.transcribe(np.asarray(audio, dtype=np.float32))
second = backend.transcribe(np.asarray(audio, dtype=np.float32))
assert first['token_ids'] == second['token_ids']
assert first['transcript'] == second['transcript']
words = set(first['transcript'].lower().split())
assert set(manifest['speech_canary']['expected_words']).issubset(words), first
long_silence = backend.transcribe(np.zeros(60 * 16_000, dtype=np.float32))
assert long_silence['audio_seconds'] == 60.0
provenance = backend.provenance()
assert provenance['backend_path'] == '/opt/voicechat-asr/nemotron_voicechat_asr_evaluator/backend.py'
assert provenance['deterministic_algorithms'] is True
assert provenance['cuda_matmul_tf32'] is False
assert provenance['cudnn_tf32'] is False
assert provenance['decoder']['attention_implementation'] == 'sdpa'
print('ASR evaluator audit passed')
PY
