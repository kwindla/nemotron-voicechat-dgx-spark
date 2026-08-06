# Weight reproduction

The normal installation path downloads the qualified Nano and EarTTS
derivatives. This procedure is the maintainer/research path for recreating them
from NVIDIA's exact public checkpoint.

## Inputs

All identities and conversion settings are frozen in
`config/conversion-recipe.json`. The command refuses a checkpoint, Nano
skeleton, Speech checkout, or replay corpus that differs from those identities.
The source checkpoint is `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` at revision
`fb0f94eaf4d03ddc430f39565229393fa1b50c26`. No earlier-access checkpoint is
accepted.

Downloaded source files are verified against the sizes and SHA-256 inventory in
`config/qualified-candidate-1.json`; conversion does not depend on optional
Hugging Face downloader cache sidecars.

The calibration release contains exact fused-embedding replay tensors: three
selected calibration conversations (3,934 Nano calls) and a physically
separate evaluation conversation (1,172 calls). It contains no microphone
recordings. The prompts in `tools/conversion/fixtures/` were authored for this
project; the replay tensors are model outputs, which OpenMDW-1.1 does not
restrict. The exhaustive inclusion/rejection decisions are in
`config/nano-gptq-corpus-selection.json`, and every payload is bound by
`config/calibration-corpus-production-candidate-1.json`.

The Hugging Face transport packages that exact replay tree as the deterministic
uncompressed `calibration/replay.tar`. Archive members are sorted and have
normalized ownership, mode, and timestamps. Bootstrap verifies the archive
against `manifests/release.json` before extraction, then verifies the extracted
tree against the per-file calibration manifest before conversion. The archive
uses index-based root names; the reproducer restores their frozen provenance
labels from the signed corpus manifest without changing any replay payload.

## One command

Inside the pinned conversion runtime, with inputs mounted read-only:

```bash
python tools/conversion/reproduce_release_artifacts.py \
  --speech-root /inputs/Speech \
  --checkpoint-root /inputs/NVIDIA-NemotronLabs-VoiceChat-11B \
  --nano-skeleton /inputs/NVIDIA-Nemotron-Nano-9B-v2 \
  --calibration-root /inputs/calibration/0 \
  --calibration-root /inputs/calibration/1 \
  --evaluation-root /inputs/evaluation/0 \
  --asr-model /inputs/parakeet \
  --work-root /outputs/voicechat-release
```

Use `--preflight-only` first to validate source identities, the complete replay
corpus, and the extraction disk budget without creating weights.

The command performs two public component extractions, two independent Nano W8
GPTQ conversions, and two independent EarTTS W8A32 conversions. It compares
every byte in each pair, applies the qualified weight-preserving runtime
configuration, and rejects any final file whose SHA-256 differs from
`config/qualified-candidate-1.json`. It then runs the held-out Nano replay gate
and the EarTTS decode/Parakeet gate in both eager and CUDA-graph modes.

The Nano replay and first-divergence implementation lives in
`nano_component_gate.py`, `nano_replay.py`, and `nano_attribution.py`.
`eartts_component_gate.py` is the standalone fixed-token audio/ASR gate. These
tools are called by the one-command flow; they can also be run directly to
localize a failed gate.

The Nano replay deliberately targets the first raw converter output because
that config exposes the full text logits required for margin comparison. The
release finalizer changes configuration only: weights are byte-identical, and
the shipped config is independently required to match the qualified hash.

## Safety and restart behavior

Every converter refuses to overwrite an output and preserves a hard floor of
100 GiB free after projected conversion output. Completed stages receive a
self-authenticating marker beneath `WORK_ROOT/stages/`. A rerun validates every
recorded file before skipping that stage. A directory without a valid marker is
treated as an incomplete conversion and is never reused implicitly; move it
aside for inspection and rerun.

Conversion reports contain diagnostic paths, timings, and environment
information and therefore are not claimed to be byte-identical. Determinism is
claimed for the complete checkpoint artifact trees and canonical release
manifests; the reports record the evidence behind those claims.

The container normalizes the final release directories to mode `0755` and
files to `0644` after all component gates pass. This changes no signed bytes and
keeps the root-container output readable to the host developer for comparison
and publication.

The only retained `ea_*` names are serialized vLLM ABI identifiers for the
qualified EarTTS W8A32 loader. They do not denote checkpoint lineage. All
source and replay validators explicitly reject earlier-access-derived inputs.

## Verified public-source reproduction

On 2026-08-05, the one-command bootstrap path completed from the pinned public
checkpoint and published replay archive on a DGX Spark. Its two independent
Nano conversions and two independent EarTTS conversions were byte-identical.
The final trees matched every Production Candidate 1 file hash and the signed
release identity
`1d8c3db8efbb87e0da3be86f1045957018082e1e8a339edaaf7e9bf4fb70671b`.

The retained reproduction report has self-hash
`ab80bbcb19c5e7ba34a1936b923480701e8b594c6fc158010d801308c9dc1e83`
and binds calibration/evaluation corpus identity
`ad8ce2fe3a66fae72ce45703216265180066b2745dc32d829ee550b45c487bd5`.
The held-out Nano gate passed all 1,172 calls with token agreement 1.0,
minimum logit cosine 0.998677, control-token order agreement 1.0, and maximum
BOS/PAD margin delta 0.3125. EarTTS eager and CUDA-graph gates both produced
the exact reference transcription with WER 0.0 under the independently hashed
Parakeet evaluator.

Bootstrap validates the report self-hash and the complete signed release before
transactionally replacing the downloaded Nano and EarTTS directories. A failed
validation restores the previously active release; a partial locally converted
tree is never exposed to `voicechat up`. If the process or machine stops in the
brief interval between directory renames, the next bootstrap restores and
verifies the signed release before removing the interrupted swap directory.

The immutable Hugging Face revision
`a20c68547d27421c1e66bd501bf295569a0cc309` remains the signed authority for
the published weight bytes. Its frozen model card predates the successful
source-reproduction and sustained-runtime campaigns. The in-repository
`release/MODEL_CARD.md` supersedes that prose with the later evidence. It was
published without changing the qualified weight payloads at immutable metadata
revision `547602dd74c934abed924c016e6546c312d8c6a3`, whose signed metadata
release identity is
`ab2a14265d0ffe2751d02f8a628c547ad95c8d6a8b7c202470b6b8597dd86e5e`.
