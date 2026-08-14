#!/usr/bin/env python3
"""Atomically patch a qualified vLLM source tree into a distinct copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

MARKER = "step5_wedge_boundary_trace"
REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "tools/provenance/qualified-deltas/index.json"
TRACE_SOURCE = Path(__file__).with_name("wedge_boundary_trace.py")
MANIFEST_NAME = "step5-wedge-diagnostic-patch-manifest.json"
QUALIFIED_TREE_SHA256 = "d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2"
QUALIFIED_FILE_COUNT = 966
TARGETS = (
    "v1/core/sched/scheduler.py",
    "v1/engine/core.py",
    "v1/engine/output_processor.py",
    "v1/worker/gpu_model_runner.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def _qualified_hashes() -> dict[str, str]:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    hashes = {entry["path"]: entry["sha256"] for entry in index["files"]}
    if set(TARGETS) - hashes.keys():
        raise ValueError("diagnostic target missing from qualified delta index")
    if len(hashes) != QUALIFIED_FILE_COUNT:
        raise ValueError(f"qualified delta index file count mismatch: {len(hashes)}")
    return hashes


def verify_qualified_source_tree(source: Path, expected: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(expected):
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"qualified source is not a regular file: {relative}")
        actual = sha256_file(path)
        if actual != expected[relative]:
            raise ValueError(f"qualified source hash mismatch: {relative}: {actual}")
        digest.update(relative.encode() + b"\0")
        digest.update(str(path.stat().st_size).encode() + b"\0")
        digest.update(actual.encode() + b"\n")
    tree_sha256 = digest.hexdigest()
    if tree_sha256 != QUALIFIED_TREE_SHA256:
        raise ValueError(f"qualified source tree mismatch: {tree_sha256}")
    return tree_sha256


def _patch_scheduler(source: str) -> str:
    source = replace_once(
        source,
        "from vllm.v1.request import Request, RequestStatus\n",
        "from vllm.v1.request import Request, RequestStatus\n"
        "from vllm.v1.wedge_boundary_trace import descriptor_key, record_boundary\n",
        "scheduler trace import",
    )
    return replace_once(
        source,
        """        self._update_after_schedule(scheduler_output)
        return scheduler_output
""",
        f"""        self._update_after_schedule(scheduler_output)

        # {MARKER}: B2 is true only after install, the complete scheduler
        # transition, construction of SchedulerOutput, and final bookkeeping.
        wedge_descriptor = {{
            "cached": list(scheduler_output.scheduled_cached_reqs.req_ids),
            "new": [item.req_id for item in scheduler_output.scheduled_new_reqs],
            "spec": scheduler_output.scheduled_spec_decode_tokens,
            "tokens": scheduler_output.num_scheduled_tokens,
        }}
        for wedge_request_id, wedge_token_count in (
            scheduler_output.num_scheduled_tokens.items()
        ):
            wedge_request = self.requests[wedge_request_id]
            record_boundary(
                "B2",
                wedge_request_id,
                custom_inputs_consumed=wedge_request.custom_inputs_num_consumed,
                custom_inputs_ready=wedge_request.custom_inputs_ready,
                descriptor_key=descriptor_key(wedge_descriptor),
                installed=wedge_request.custom_inputs is not None,
                proposal_token_ids=list(wedge_request.spec_token_ids),
                request_status=str(wedge_request.status),
                running=wedge_request in self.running,
                token_count=int(wedge_token_count),
                waiting_input=wedge_request_id in self.waiting_input,
            )
        return scheduler_output
""",
        "scheduler B2",
    )


def _patch_core(source: str) -> str:
    source = replace_once(
        source,
        "from vllm.v1.engine.utils import (\n",
        "from vllm.v1.wedge_boundary_trace import record_boundary\n"
        "from vllm.v1.engine.utils import (\n",
        "core trace import",
    )
    source = replace_once(
        source,
        """        elif request_type == EngineCoreRequestType.APPEND:
            request_id, custom_inputs = request
            self.set_custom_inputs(request_id, custom_inputs)
""",
        f"""        elif request_type == EngineCoreRequestType.APPEND:
            request_id, custom_inputs = request
            # {MARKER}: B1 follows actual request-type decoding.
            record_boundary("B1", request_id, decoded=True)
            self.set_custom_inputs(request_id, custom_inputs)
""",
        "core B1",
    )
    source = replace_once(
        source,
        """        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
""",
        f"""        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
            # {MARKER}: B6 is true only after put_nowait succeeds.
            for item in output[1].outputs:
                record_boundary("B6", item.request_id, enqueued=True)
""",
        "core B6",
    )
    return replace_once(
        source,
        """                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
""",
        f"""                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                # {MARKER}: returned tracker means send accepted the message.
                for item in outputs.outputs:
                    record_boundary("B7", item.request_id, sent=True)
""",
        "core B7",
    )


def _patch_output_processor(source: str) -> str:
    source = replace_once(
        source,
        "from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason\n",
        "from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason\n"
        "from vllm.v1.wedge_boundary_trace import record_boundary\n",
        "output processor trace import",
    )
    source = replace_once(
        source,
        """        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)
""",
        f"""        for request_id, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)
            # {MARKER}: propagated task failure was published to collector.
            record_boundary(
                "B8", request_id, exception_kind=type(e).__name__,
                received=True, routed=True
            )
""",
        "output processor error B8",
    )
    source = replace_once(
        source,
        """                    req_state.queue.put(request_output)
            elif parent := self.parent_requests.get(request_id):
""",
        f"""                    req_state.queue.put(request_output)
                    # {MARKER}: abort output was published to collector.
                    record_boundary(
                        "B8", request_id, cancellation_kind="abort",
                        received=True, routed=True
                    )
            elif parent := self.parent_requests.get(request_id):
""",
        "output processor abort B8",
    )
    return replace_once(
        source,
        """                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
""",
        f"""                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                    # {MARKER}: B8 follows successful collector publication.
                    record_boundary(
                        "B8", req_id, received=True, routed=True
                    )
""",
        "output processor B8",
    )


def _patch_runner(source: str) -> str:
    source = replace_once(
        source,
        "from vllm.v1.worker.utils import is_residual_scattered_for_sp\n",
        "from vllm.v1.worker.utils import is_residual_scattered_for_sp\n"
        "from vllm.v1.wedge_boundary_trace import descriptor_key, record_boundary\n",
        "runner trace import",
    )
    source = replace_once(
        source,
        """        # Run the model.
        # Use persistent buffers for CUDA graphs.
""",
        f"""        # {MARKER}: B3 observes final post-override dispatch values.
        wedge_descriptor_key = descriptor_key({{
            "batch_descriptor": batch_descriptor,
            "graph_mode": str(cudagraph_runtime_mode),
            "token_count": int(num_input_tokens),
        }})
        for wedge_request_id in self.input_batch.req_ids:
            record_boundary(
                "B3",
                wedge_request_id,
                batch_size=int(self.input_batch.num_reqs),
                descriptor_key=wedge_descriptor_key,
                epoch=getattr(self, "engine_index", 0),
                graph_mode=str(cudagraph_runtime_mode),
                token_count=int(num_input_tokens),
            )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
""",
        "runner B3",
    )
    source = replace_once(
        source,
        """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("Postprocess"):
""",
        f"""            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        # {MARKER}: B4 follows the actual forward return.
        for wedge_request_id in self.input_batch.req_ids:
            record_boundary(
                "B4", wedge_request_id, descriptor_key=wedge_descriptor_key
            )

        with record_function_or_nullcontext("Postprocess"):
""",
        "runner B4",
    )
    return replace_once(
        source,
        """            self._update_states_after_model_execute(output_token_ids)

        return sampler_output
""",
        f"""            self._update_states_after_model_execute(output_token_ids)

        # {MARKER}: B5 follows sampling, rejection, and state commit/restore.
        for wedge_request_id in self.input_batch.req_ids:
            record_boundary(
                "B5",
                wedge_request_id,
                commit_outcome="post_execute_state_updated",
                proposed_count=(
                    0 if spec_decode_metadata is None else
                    int(spec_decode_metadata.num_draft_tokens.sum())
                ),
                sampled=True,
                speculative=spec_decode_metadata is not None,
            )

        return sampler_output
""",
        "runner B5",
    )


PATCHERS: dict[str, Callable[[str], str]] = {
    "v1/core/sched/scheduler.py": _patch_scheduler,
    "v1/engine/core.py": _patch_core,
    "v1/engine/output_processor.py": _patch_output_processor,
    "v1/worker/gpu_model_runner.py": _patch_runner,
}


def _assert_separate_roots(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    output = output.resolve(strict=False)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output roots must be distinct and unnested")
    if output.exists():
        raise FileExistsError(f"output root must not exist: {output}")
    source_stat = source.stat()
    if output.parent.exists():
        parent_stat = output.parent.stat()
        if (source_stat.st_dev, source_stat.st_ino) == (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ):
            raise ValueError("output root aliases the qualified source root")


def _compile_tree(root: Path) -> None:
    for path in sorted(root.rglob("*.py")):
        compile(path.read_bytes(), str(path), "exec")


def apply_patch(
    qualified_source_root: Path,
    output_root: Path,
    *,
    failure_point: str | None = None,
) -> dict[str, object]:
    """Verify, stage, compile, and atomically publish without source writes."""
    source = qualified_source_root.resolve(strict=True)
    output = output_root.resolve(strict=False)
    _assert_separate_roots(source, output)
    expected = _qualified_hashes()

    # Verify all 966 source hashes, then every patch anchor, before the first write.
    verified_tree_sha256 = verify_qualified_source_tree(source, expected)
    before: dict[str, str] = {}
    patched_text: dict[str, str] = {}
    for relative in TARGETS:
        path = source / relative
        before[relative] = expected[relative]
        patched_text[relative] = PATCHERS[relative](path.read_text(encoding="utf-8"))
    if failure_point == "after_verify":
        raise RuntimeError("injected patcher failure after verification")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=True)
        verify_qualified_source_tree(staging, expected)
        if failure_point == "after_copy":
            raise RuntimeError("injected patcher failure after copy")
        after: dict[str, str] = {}
        for index, relative in enumerate(TARGETS):
            path = staging / relative
            path.write_text(patched_text[relative], encoding="utf-8")
            after[relative] = sha256_file(path)
            if failure_point == f"after_target_{index}":
                raise RuntimeError(f"injected patcher failure after target {index}")
        trace_target = staging / "v1/wedge_boundary_trace.py"
        shutil.copyfile(TRACE_SOURCE, trace_target)
        after["v1/wedge_boundary_trace.py"] = sha256_file(trace_target)
        if failure_point == "before_compile":
            raise RuntimeError("injected patcher failure before compile")
        _compile_tree(staging)
        if failure_point == "after_compile":
            raise RuntimeError("injected patcher failure after compile")
        manifest: dict[str, object] = {
            "baseline_tree_sha256": verified_tree_sha256,
            "before": before,
            "after": after,
            "marker": MARKER,
            "patcher_sha256": sha256_file(Path(__file__)),
            "trace_source_sha256": sha256_file(TRACE_SOURCE),
            "verified_source_file_count": len(expected),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if failure_point == "before_publish":
            raise RuntimeError("injected patcher failure before publish")
        os.replace(staging, output)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            apply_patch(
                args.qualified_source_root,
                args.output_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
