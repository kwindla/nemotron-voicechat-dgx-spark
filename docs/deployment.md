# Deployment and lifecycle

## Supported host

Production Candidate 1 is qualified on DGX Spark (`linux/aarch64`, NVIDIA GB10)
with driver 580.142 and kernel 6.17.0-1014-nvidia. Install Docker with NVIDIA GPU
support, `uv`, `git`, and `curl`. Run `./voicechat doctor` for an explicit report.
A different driver, kernel, GPU, or architecture is a new qualification target.

## Bootstrap

```bash
./voicechat bootstrap
```

Bootstrap is resumable and content-addressed. It synchronizes the pinned Python
3.12.13 host environment, downloads the converted release and its exact public
parents by immutable Hugging Face revision, verifies every declared SHA-256,
downloads the pinned Pocket TTS snapshots, builds the public CUDA image, audits
its lineage/SBOM, and records state under `~/.cache/nemotron-voicechat`.
Mutable traces live under `~/.local/state/nemotron-voicechat/traces`. Override
these locations with the global `--cache-root` and `--trace-root` options or
their `VOICECHAT_CACHE_ROOT`/`VOICECHAT_TRACE_ROOT` process variables.

HF credentials are only download-process inputs. For a gated 401/403, accept
the terms at the URL printed by bootstrap, authenticate with `hf auth login` or
set `HF_TOKEN` for that command, then retry. No credential is copied into the
image or application configuration.

After one successful online bootstrap, this performs a network-free cache and
lock validation:

```bash
./voicechat bootstrap --offline
```

## Foreground stack

```bash
./voicechat up
```

One foreground launcher starts the Docker model, waits up to 15 minutes for
model and Pocket-worker readiness, starts the host Pipecat bot, prints the
Playground URL, and owns teardown. Ctrl-C stops Pipecat and removes the
`--rm` model container. If the launcher was killed, `./voicechat down` removes
only its deterministic orphan container.

The raw model WebSocket is published only as `127.0.0.1:8786`; the browser sees
Pipecat at `127.0.0.1:7860`. To serve a LAN deliberately:

```bash
./voicechat up --host 0.0.0.0 --port 7860
```

This changes only the Pipecat listener. The raw model port remains bound to
`127.0.0.1` regardless of the Pipecat `--host` value.

For an optional tunnel, invoke ngrok yourself against Pipecat, never the model:

```bash
ngrok http http://127.0.0.1:7860
```

There is no system service, Compose file, `.env`, or background daemon managed
by this repository.

## Configuration

`config/production-candidate-1.toml` pins artifacts, image names, host profile,
ports, and the full qualified model environment. The launcher materializes
those values as explicit Docker environment entries. Cache roots, trace roots,
and listening ports are deployment values; changing model/turn/latency values
creates a different candidate.

The model container mounts only verified model roots and Pocket's HF cache
read-only; traces are the sole read-write mount. NVIDIA Speech and the project
server are built into the image. `VLLM_ALLOW_INSECURE_SERIALIZATION` is unset.
The codec worker is pinned to cores 5–6 and Pocket TTS to 7–9 and 15. Pocket's
`/health` object echoes its effective `sched_getaffinity` CPU list; operators
should treat overlap with codec cores as a startup/configuration defect.

## Development and release

```bash
./voicechat test
uv run --frozen ruff check .
```

`./voicechat test --live` includes DGX/browser integration and is intentionally
expensive; it generates its own speech fixtures and only requires `--asr-model`.
Its default run exceeds 15 minutes and verifies graceful session-limit closure.
See `docs/qualification.md` for its ordered gates and evidence layout.
The initial public release downloads the signed converted artifacts. The
maintainer publication tooling and deterministic production conversion
pipeline will follow in stage 2. The public build graph and current
reproduction status are documented in `docs/public-build-graph.md` and
`docs/weight-reproduction.md`.
