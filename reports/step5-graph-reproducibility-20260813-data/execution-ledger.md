# Step 5 execution ledger

All timestamps embedded in raw stdout are UTC. Artifact filenames retain the
user-requested `20260813` study date even where execution continued after UTC
midnight.

## Fixed environment

- image: `pipecat-ai/nemotron-voicechat-dgx-spark:generation-step2`
- Speech root: `/opt/Speech`
- production Nano: `/models/derived/nano`
- retained capture: `/capture`
- `HF_MODULES_CACHE=/tmp/voicechat-hf-modules`
- `HF_HOME=/models/huggingface`
- `HUGGINGFACE_HUB_CACHE=/models/huggingface/hub`
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- container flags: `--rm --init --gpus all --ipc host --shm-size 16g
  --security-opt label=disable`
- standard tool arguments: `--speech-root /opt/Speech --capture-root /capture
  --reference-outputs /results/reference-output-digests.json --sampling-seed 0`

Every engine label ran in its own fresh Docker container. `run/*.stdout.log`
contains the complete engine configuration emitted by vLLM, including graph
mode and engine seed. The matching JSON contains the full tool arguments that
affect inference, artifact/capture hashes, token vectors, mismatch records,
cleanup status, and output digests.

## Arms in execution order

1. `g0-1` initial attempt: graph, calls 0–46, three blocks. Retained as
   `g0-1-superseded-digest-encoder.*`; token evidence valid, interface digest
   encoder superseded.
2. `g0-1`, `g0-2`, `g0-3`: graph, seed 0, calls 0–46, three blocks each.
3. `e0-1`, `e0-2`, `e0-3`: eager, seed 0, calls 0–46, three blocks each.
4. `g1-1`, `g1-2`: graph, sampling seed 1, calls 0–46, three blocks each.
5. `history-1` initial attempt: graph, baseline then 47-call and 17-call
   histories, seven blocks. Retained as
   `history-1-superseded-aggregate-count.*`; observations valid, report-level
   count check superseded.
6. `history-1`, `history-2`: same history sequence, seven blocks each.
7. `schedule-1`, `schedule-2`: graph, three unique-ID blocks, three reused-ID
   blocks, and three unique-ID blocks with CUDA synchronization after every
   call.
8. Diagnostic overlay construction using
   `tools.conversion.nano_component_gate.prepare_diagnostic_candidate`; provenance
   is `diagnostic-build-provenance.json`.
9. `diag-g-1`, `diag-g-2`, `diag-g-3`: graph diagnostic overlay, calls 0–46,
   three blocks, bounded production-sampler top-20 capture at call 39.
10. `diag-e-1`: corresponding eager diagnostic overlay.
11. `confirm-graph-1` through `confirm-graph-3`: production graph, calls 0–686,
    three blocks per engine.
12. `confirm-eager-1` through `confirm-eager-3`: production eager, calls 0–686,
    three blocks per engine.

## Service operations

Pre-stop health and inspect are under `preflight/`. The model container was
stopped with `docker stop --time 30 nemotron-voicechat-model`; the stopped
container was removed only to free its fixed name. Restoration invoked exactly:

```text
bash /tmp/claude-1000/-home-khkramer-src-nemotron-voicechat-dgx-spark/9dab1982-03af-4f77-9d25-ddf7139af66f/scratchpad/start-step2-container-v4.sh
```

No image or volume was removed or rebuilt. The first restoration startup stalled
at the first EarTTS pipeline-warm-up request and is fully retained. The same
command and image were used for the restoration retry.

## Validation commands

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/benchmark/graph_reproducibility.py tests/runtime/test_graph_reproducibility.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q tests/runtime/test_graph_reproducibility.py tests/runtime/test_nano_interface_decomposition.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/qualification/stratified_latency_analyzer.py verify-anchors --output reports/step5-graph-reproducibility-20260813-data/validation-anchors.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -m 'not browser_e2e'
```
