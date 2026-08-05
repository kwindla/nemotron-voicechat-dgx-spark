#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: audit-public-runtime.sh IMAGE [OUTPUT_DIRECTORY]}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${2:-${repo_root}/artifacts/public-runtime-audit}
mkdir -p "${output}"

docker image inspect "${image}" >"${output}/image-inspect.json"
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
docker run --rm -i --network none "${image}" /opt/pocket-tts/bin/python - <<'PY' \
  >"${output}/pocket-runtime-identity.json"
import json
import torch
print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda}))
assert torch.version.cuda is None
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
