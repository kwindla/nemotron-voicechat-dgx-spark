#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: audit-public-runtime.sh IMAGE [OUTPUT_DIRECTORY]}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${2:-${repo_root}/artifacts/public-runtime-audit}
mkdir -p "${output}"
read -r project_source_sha256 project_payload_sha256 build_recipe_sha256 recipe_snapshot_sha256 < <(
  PYTHONPATH="${repo_root}/src" python3 -c \
    'import hashlib, sys; from pathlib import Path; from nemotron_voicechat_runtime.public_runtime_identity import public_runtime_payload_sha256, public_runtime_recipe_sha256, public_runtime_recipe_snapshot, public_runtime_source_sha256; root=Path(sys.argv[1]); print(public_runtime_source_sha256(root), public_runtime_payload_sha256(root), public_runtime_recipe_sha256(root), hashlib.sha256(public_runtime_recipe_snapshot(root)).hexdigest())' \
    "${repo_root}"
)

docker image inspect "${image}" >"${output}/image-inspect.json"
python3 - "${output}/image-inspect.json" \
  "${project_source_sha256}" "${project_payload_sha256}" "${build_recipe_sha256}" <<'PY'
import json
import sys

path, expected_source, expected_payload, expected_recipe = sys.argv[1:]
inspect = json.load(open(path, encoding="utf-8"))[0]
labels = inspect.get("Config", {}).get("Labels") or {}
expected = {
    "ai.pipecat.voicechat.project-source-sha256": expected_source,
    "ai.pipecat.voicechat.project-payload-sha256": expected_payload,
    "ai.pipecat.voicechat.build-recipe-sha256": expected_recipe,
}
actual = {key: labels.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"runtime image source labels mismatch: image={actual!r} checkout={expected!r}")
PY
docker history --no-trunc "${image}" >"${output}/docker-history.txt"
lineage_scan_status=0
grep -E -i 'voicechat-vllm:ea-|ea-nemotron|public-candidate1-hotfix|public-runtime-flat' \
  "${output}/image-inspect.json" "${output}/docker-history.txt" \
  || lineage_scan_status=$?
if [[ ${lineage_scan_status} -eq 0 ]]; then
  echo "Prohibited historical image lineage found" >&2
  exit 2
elif [[ ${lineage_scan_status} -ne 1 ]]; then
  echo "Runtime image lineage scan failed with status ${lineage_scan_status}" >&2
  exit 2
fi

docker run --rm -i --network none \
  -v "${repo_root}/tools/runtime/generate_runtime_sbom.py:/audit/generate_runtime_sbom.py:ro" \
  "${image}" python3 /audit/generate_runtime_sbom.py >"${output}/sbom.spdx.json"
python3 - "${output}/sbom.spdx.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["spdxVersion"] == "SPDX-2.3"
assert payload["packages"]
PY
docker run --rm -i --network none "${image}" python3 - <<'PY' >"${output}/runtime-identity.json"
import importlib.metadata
import json
import os
import platform
import torch
import vllm

assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ
print(json.dumps({
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "transformers": importlib.metadata.version("transformers"),
}, indent=2, sort_keys=True))
PY
docker run --rm -i --network none \
  -e PYTHONPATH=/opt/Speech \
  "${image}" python3 - <<'PY' \
  >"${output}/eartts-request-generation-canary.json"
import asyncio
import contextlib
import io
import json

noise = io.StringIO()
with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
    from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import (
        LLMStreamingEngine,
        StreamStatus,
    )
    from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import (
        StreamingS2SPipeline,
    )


class Backend:
    def __init__(self):
        self.aborts = []
        self.live = set()

    async def abort(self, request_id):
        self.aborts.append(request_id)
        self.live.discard(request_id)


class FailingBackend(Backend):
    async def abort(self, request_id):
        raise RuntimeError(f"forced abort failure for {request_id}")


class Iterator:
    def __init__(self, request_id, backend):
        self.request_id = request_id
        self.backend = backend
        self.closed = False

    async def aclose(self):
        self.closed = True
        await self.backend.abort(self.request_id)


async def main():
    engine = object.__new__(LLMStreamingEngine)
    engine.engine = Backend()
    engine.requests = {}
    engine._unique_backend_request_ids = True
    engine._backend_request_generation = 0

    await engine.start_generation("logical")
    old = engine.requests["logical"]
    old_id = old.backend_request_id
    engine.engine.live.add(old_id)
    old.generation_iterator = Iterator(old_id, engine.engine)
    assert await engine.abort_generation("logical") is True
    assert old.generation_iterator.closed is True

    await engine.start_generation("logical")
    current = engine.requests["logical"]
    current_id = current.backend_request_id
    engine.engine.live.add(current_id)
    assert old_id != current_id
    assert current.backend_generation == 2
    assert engine.resolve_backend_request_id("logical") == current_id

    # Reproduce the original hazard: cleanup captured by the old generator
    # arrives after the replacement is live.  Its physical ID must be harmless.
    await engine.engine.abort(old_id)
    assert current_id in engine.engine.live
    assert current.status is StreamStatus.ACTIVE

    failed = object.__new__(LLMStreamingEngine)
    failed.engine = FailingBackend()
    failed.requests = {}
    failed._unique_backend_request_ids = True
    failed._backend_request_generation = 0
    await failed.start_generation("logical")
    failed_state = failed.requests["logical"]
    failed_state.generation_iterator = Iterator(
        failed_state.backend_request_id, failed.engine
    )
    try:
        await failed.start_generation("logical")
    except RuntimeError as exc:
        assert "Could not retire" in str(exc)
    else:
        raise AssertionError("failed retirement overwrote the active generation")
    assert failed.requests["logical"] is failed_state
    assert failed._backend_request_generation == 1

    inactive = object.__new__(LLMStreamingEngine)
    inactive.engine = Backend()
    inactive.requests = {}
    inactive._unique_backend_request_ids = True
    inactive._backend_request_generation = 0
    await inactive.start_generation("logical")
    inactive_state = inactive.requests["logical"]
    inactive_iterator = Iterator(inactive_state.backend_request_id, inactive.engine)
    inactive_state.generation_iterator = inactive_iterator
    inactive_state.status = StreamStatus.FINISHED
    assert await inactive.abort_generation("logical") is True
    assert inactive_iterator.closed is True
    assert "logical" not in inactive.requests

    invalid = object.__new__(LLMStreamingEngine)
    invalid.engine = Backend()
    invalid.requests = {}
    invalid._unique_backend_request_ids = True
    invalid._backend_request_generation = 0
    await invalid.start_generation("logical")
    invalid.requests["logical"].backend_request_id = "logical"
    try:
        invalid.resolve_backend_request_id("logical")
    except RuntimeError as exc:
        assert "identity is invalid" in str(exc)
    else:
        raise AssertionError("invalid physical identity resolved to the logical ID")

    default_engine = object.__new__(LLMStreamingEngine)
    default_engine.engine = Backend()
    default_engine.requests = {}
    default_engine._unique_backend_request_ids = False
    default_engine._backend_request_generation = 0
    await default_engine.start_generation("default")
    assert default_engine.resolve_backend_request_id("default") == "default"
    assert default_engine.requests["default"].backend_generation == 0
    default_iterator = Iterator("default", default_engine.engine)
    default_engine.requests["default"].generation_iterator = default_iterator
    assert await default_engine.abort_generation("default") is True
    assert default_iterator.closed is False

    direct_before = {
        "status": "active",
        "iterator_id": 11,
        "backend_request_id": current_id,
        "backend_generation": 2,
    }
    StreamingS2SPipeline._validate_direct_request_continuity(
        "EarTTS", direct_before, dict(direct_before)
    )
    for key, value in (
        ("backend_request_id", old_id),
        ("backend_generation", 1),
    ):
        mutated = dict(direct_before)
        mutated[key] = value
        try:
            StreamingS2SPipeline._validate_direct_request_continuity(
                "EarTTS", direct_before, mutated
            )
        except RuntimeError as exc:
            assert "during direct position" in str(exc)
        else:
            raise AssertionError(f"direct-position mutation passed: {key}")

    return {
        "schema": 1,
        "logical_request_id": "logical",
        "old_backend_request_id": old_id,
        "current_backend_request_id": current_id,
        "current_generation": current.backend_generation,
        "stale_abort_isolated": True,
        "iterator_closed_before_reuse": True,
        "failed_retirement_preserves_generation": True,
        "inactive_iterator_closed": True,
        "invalid_identity_rejected": True,
        "default_identity_mapping": True,
        "default_iterator_lifecycle_unchanged": True,
        "direct_position_generation_mutations_rejected": True,
    }


with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
    result = asyncio.run(main())
print(json.dumps(result, indent=2, sort_keys=True))
PY
docker run --rm -i --network none "${image}" /opt/pocket-tts/bin/python - <<'PY' \
  >"${output}/pocket-runtime-identity.json"
import json
import torch
print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda}))
assert torch.version.cuda is None
PY
docker run --rm -i --network none "${image}" python3 - \
  "${project_payload_sha256}" "${build_recipe_sha256}" "${recipe_snapshot_sha256}" \
  >"${output}/copied-source-identity.json" <<'PY'
import hashlib
import json
import pathlib
import sys

from nemotron_voicechat_runtime.public_runtime_identity import public_runtime_payload_sha256

expected_payload, expected_recipe, expected_snapshot = sys.argv[1:]
project_root = pathlib.Path("/opt/project")
snapshot = project_root / "build-recipe/recipe-snapshot.json"
actual_payload = public_runtime_payload_sha256(project_root)
actual_snapshot = hashlib.sha256(snapshot.read_bytes()).hexdigest()
if actual_payload != expected_payload or actual_snapshot != expected_snapshot:
    raise SystemExit("runtime copied source payload/recipe snapshot differs from checkout")
print(json.dumps({
    "payload_sha256": actual_payload,
    "recipe_sha256": expected_recipe,
    "recipe_snapshot_sha256": actual_snapshot,
}, indent=2, sort_keys=True))
PY

python3 - "${image}" "${output}" <<'PY'
import hashlib
import json
import pathlib
import sys

image, output_arg = sys.argv[1:]
output = pathlib.Path(output_arg)
inspect = json.loads((output / "image-inspect.json").read_text())[0]
evidence = {}
for path in sorted(output.iterdir()):
    if path.name == "provenance.json" or not path.is_file():
        continue
    evidence[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": 1,
    "kind": "nemotron_voicechat_public_runtime_attestation",
    "image": image,
    "image_id": inspect["Id"],
    "architecture": inspect["Architecture"],
    "labels": inspect.get("Config", {}).get("Labels") or {},
    "repo_digests": inspect.get("RepoDigests") or [],
    "evidence_sha256": evidence,
}
(output / "provenance.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "Public runtime audit written to ${output}"
