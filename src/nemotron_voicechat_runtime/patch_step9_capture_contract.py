#!/usr/bin/env python3
"""Install the hash-qualified Step-9 capture inventory and strict-ready gate."""

from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path

TARGETS = {
    "dispatcher": (
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/"
            "cudagraph_dispatcher.py"
        ),
        {
            "76f9530a9800c6ad971cbc789ce94461a7395d0221bff00fb916ccf3fa9f684c",
            # production-candidate-2, after the v1 Step-9 contract was installed.
            "35b89d55e3c23959f43dc697ba11c612a90fc5f52032bc4587b25d62156a90cc",
        },
    ),
    "backends": (
        Path("/usr/local/lib/python3.12/dist-packages/vllm/compilation/backends.py"),
        {
            "8cd2f8edfa082555bd71a2ab7626cbb15b2e02f4ef0754dccdde7254ed2da5e0",
            # production-candidate-2, after the v1 Step-9 contract was installed.
            "b21e4eb67af555506119282a63eab95b96b80b812e7cb67efe370bbf1ecc3750",
        },
    ),
}
OLD_MARKER = "voicechat_step9_capture_contract_v1"
MARKER = "voicechat_step9_capture_contract_v2"


def qualified_source(name: str) -> tuple[Path, str]:
    path, expected_hashes = TARGETS[name]
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual not in expected_hashes:
        raise RuntimeError(
            f"unqualified input {path}: {actual} not in {sorted(expected_hashes)}"
        )
    return path, data.decode("utf-8")


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def patch_dispatcher(source: str) -> str:
    if OLD_MARKER in source:
        source = replace_once(
            source,
            "from vllm.forward_context import BatchDescriptor\n",
            "from vllm.forward_context import BatchDescriptor\n"
            "from vllm.logger import init_logger\n\n"
            "logger = init_logger(__name__)\n",
            "dispatcher logger",
        )
        source = replace_once(
            source,
            f'''        # {OLD_MARKER}: reduced-capture promotion forbids an uncovered
        # post-ready shape. Fail closed instead of silently entering eager mode.
        if (
            os.environ.get("VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE", "0") == "1"
            and os.path.exists("/tmp/voicechat-step9-ready")
        ):
            raise RuntimeError(
                "Step-9 uncovered post-ready cudagraph descriptor: %r"
                % (batch_descriptor,)
            )

''',
            _coverage_fallback_source(),
            "v1 strict post-ready coverage",
        )
        return source.replace(OLD_MARKER, MARKER)

    source = replace_once(
        source,
        "import os\n",
        "import json\nimport os\n",
        "dispatcher json import",
    )
    source = replace_once(
        source,
        "from vllm.forward_context import BatchDescriptor\n",
        "from vllm.forward_context import BatchDescriptor\n"
        "from vllm.logger import init_logger\n\n"
        "logger = init_logger(__name__)\n",
        "dispatcher logger",
    )
    source = replace_once(
        source,
        """        self.keys_initialized = False
""",
        f"""        self.keys_initialized = False
        # {MARKER}: inventory each distinct runtime descriptor without changing
        # dispatch. This is enabled only for the baseline inventory arm.
        self.voicechat_step9_inventory_seen = set()
""",
        "inventory state",
    )
    source = replace_once(
        source,
        """        # if not initialized, just skip dispatching.
""",
        f"""        # {MARKER}: record exact descriptors before any rounding to a
        # captured key. The ready bit separates startup warmup from client work.
        if os.environ.get("VOICECHAT_STEP9_SHAPE_INVENTORY", "0") == "1":
            item = (
                int(batch_descriptor.num_tokens),
                bool(batch_descriptor.uniform_decode),
                int(getattr(batch_descriptor, "query_len", 0)),
                bool(use_cascade_attn),
                os.path.exists("/tmp/voicechat-step9-ready"),
            )
            if item not in self.voicechat_step9_inventory_seen:
                self.voicechat_step9_inventory_seen.add(item)
                print(
                    "VOICECHAT_STEP9_SHAPE "
                    + json.dumps(
                        {{
                            "num_tokens": item[0],
                            "uniform_decode": item[1],
                            "query_len": item[2],
                            "use_cascade_attn": item[3],
                            "post_ready": item[4],
                        }},
                        sort_keys=True,
                    ),
                    flush=True,
                )

        # if not initialized, just skip dispatching.
""",
        "inventory hook",
    )
    return replace_once(
        source,
        """        # finally, just return no cudagraphs
        return CUDAGraphMode.NONE, None
""",
        _coverage_fallback_source()
        + """
        # finally, just return no cudagraphs
        return CUDAGraphMode.NONE, None
""",
        "strict post-ready coverage",
    )


def _coverage_fallback_source() -> str:
    return f'''        # {MARKER}: uncovered post-ready descriptors use vLLM's
        # graphless path in production. Qualification can opt into a hard
        # coverage assertion; neither path triggers a new graph capture/compile.
        if os.path.exists("/tmp/voicechat-step9-ready"):
            if os.environ.get("VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE", "0") == "1":
                raise RuntimeError(
                    "Step-9 uncovered post-ready cudagraph descriptor: %r"
                    % (batch_descriptor,)
                )
            fallback_record = json.dumps(
                {{
                    "num_tokens": int(batch_descriptor.num_tokens),
                    "uniform_decode": bool(batch_descriptor.uniform_decode),
                    "query_len": int(getattr(batch_descriptor, "query_len", 0)),
                    "pid": os.getpid(),
                }},
                sort_keys=True,
            ) + "\\n"
            fallback_fd = os.open(
                "/tmp/voicechat-step9-fallbacks.jsonl",
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(fallback_fd, fallback_record.encode("utf-8"))
            finally:
                os.close(fallback_fd)
            logger.warning_once(
                "Step-9 uncovered post-ready cudagraph descriptor: %r; "
                "falling back to graphless execution",
                batch_descriptor,
            )

'''


def patch_backends(source: str) -> str:
    if OLD_MARKER in source:
        return source.replace(OLD_MARKER, MARKER)
    return replace_once(
        source,
        """    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        vllm_config = self.vllm_config
""",
        f"""    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        # {MARKER}: compilation after readiness is a release-blocking failure.
        if (
            os.environ.get("VOICECHAT_STEP9_ASSERT_NO_POST_READY_COMPILE", "0") == "1"
            and os.path.exists("/tmp/voicechat-step9-ready")
        ):
            raise RuntimeError("Step-9 detected torch.compile after application readiness")
        vllm_config = self.vllm_config
""",
        "post-ready compile assertion",
    )


def main() -> None:
    patchers = {
        "dispatcher": patch_dispatcher,
        "backends": patch_backends,
    }
    for name, patcher in patchers.items():
        path, source = qualified_source(name)
        patched = patcher(source)
        if MARKER not in patched:
            raise RuntimeError(f"{name}: patch marker missing")
        path.write_text(patched, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)
        print(f"Installed Step-9 capture contract in {path}")


if __name__ == "__main__":
    main()
