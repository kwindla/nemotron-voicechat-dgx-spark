#!/usr/bin/env python3
"""Emit a deterministic SPDX 2.3 inventory for the runtime container."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess


def spdx_id(kind: str, name: str, version: str) -> str:
    value = re.sub(r"[^A-Za-z0-9.-]+", "-", f"{kind}-{name}-{version}")
    return f"SPDXRef-{value.strip('-')}"


def package(name: str, version: str, kind: str) -> dict[str, object]:
    return {
        "SPDXID": spdx_id(kind, name, version),
        "name": name,
        "versionInfo": version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "supplier": "NOASSERTION",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:{kind}/{name}@{version}",
            }
        ],
    }


def main() -> None:
    packages: dict[str, dict[str, object]] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or distribution.name).lower()
        item = package(name, distribution.version, "pypi")
        packages[str(item["SPDXID"])] = item

    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        name, version = line.split("\t", 1)
        item = package(name, version, "deb")
        packages[str(item["SPDXID"])] = item

    ordered = [packages[key] for key in sorted(packages)]
    inventory = "\n".join(f"{item['SPDXID']}\t{item['versionInfo']}" for item in ordered).encode()
    namespace_hash = hashlib.sha256(inventory).hexdigest()
    document_id = "SPDXRef-DOCUMENT"
    payload = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": document_id,
        "name": "nemotron-voicechat-dgx-spark-runtime",
        "documentNamespace": (
            "https://pipecat.ai/spdx/nemotron-voicechat-dgx-spark/" + namespace_hash
        ),
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: tools/runtime/generate_runtime_sbom.py"],
        },
        "packages": ordered,
        "relationships": [
            {
                "spdxElementId": document_id,
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": str(item["SPDXID"]),
            }
            for item in ordered
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
