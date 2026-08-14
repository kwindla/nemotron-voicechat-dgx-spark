#!/usr/bin/env python3
"""Run the v2 CPU Step 5 protocol-race campaign on real asyncio task paths."""

from __future__ import annotations

import argparse
import ast
import asyncio
import enum
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from nemotron_voicechat_runtime.patch_step5_wedge_boundary_trace import (
    apply_patch,
)
from nemotron_voicechat_runtime.wedge_boundary_trace import (
    BOUNDARIES,
    close_trace,
    descriptor_key,
    record_boundary,
    trace_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "tools/provenance/qualified-deltas/index.json"
DESIGN_PATH = REPO_ROOT / "docs/step5-pairing-wedge-closure-design.md"
PREREG_PATH = REPO_ROOT / "reports/step5-wedge/preregistration-v3.md"
QUALIFIED_TREE_SHA256 = "d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2"
APPLICATION_REVISION = "7e2e47fddb0da8288651d3158fc5a93b46ca202e"
ROOT_SEED = 0x5EED5A17B0B9C0DE
MASK64 = (1 << 64) - 1
SCHEDULE_VERSION = 3
REQUIRED_SCHEDULES = 1_000_000
REQUIRED_DISTINCT_INTERLEAVINGS = 64
DEFAULT_BATCH_SIZE = 10_000
DEFAULT_CONCURRENCY = 128
YIELD_POINTS = tuple(range(24))
HARNESS_FILES = (
    "src/nemotron_voicechat_runtime/wedge_boundary_trace.py",
    "src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py",
    "src/nemotron_voicechat_runtime/runtime_optimizations.py",
    "tools/qualification/step5_phase0_protocol_race.py",
    "tests/runtime/test_step5_wedge_boundary_trace.py",
    "tests/runtime/test_step5_phase0_protocol_race.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def domain_seed(domain: str) -> int:
    domain_word = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    return splitmix64(ROOT_SEED ^ domain_word)


def verify_application_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if revision != APPLICATION_REVISION:
        raise ValueError(
            f"application revision mismatch: expected {APPLICATION_REVISION}, got {revision}"
        )
    return revision


def verify_qualified_tree(root: Path) -> dict[str, Any]:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for entry in sorted(index["files"], key=lambda item: item["path"]):
        path = root / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"qualified vLLM source mismatch: {entry['path']}")
        actual = sha256_file(path)
        digest.update(entry["path"].encode() + b"\0")
        digest.update(str(path.stat().st_size).encode() + b"\0")
        digest.update(actual.encode() + b"\n")
    if digest.hexdigest() != QUALIFIED_TREE_SHA256:
        raise ValueError(f"qualified vLLM tree mismatch: {digest.hexdigest()}")
    return {
        "files": index["qualified_python_files"],
        "index_sha256": sha256_file(INDEX_PATH),
        "root": str(root),
        "tree_sha256": digest.hexdigest(),
    }


class _MsgspecStruct:
    def __init_subclass__(cls, **_kwargs: Any) -> None:
        super().__init_subclass__()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        names = list(self.__class__.__annotations__)
        values = dict(zip(names, args, strict=False))
        values.update(kwargs)
        for name in names:
            if name in values:
                value = values[name]
            elif hasattr(self.__class__, name):
                default = getattr(self.__class__, name)
                value = default.copy() if isinstance(default, (dict, list, set)) else default
            else:
                raise TypeError(f"missing required field: {name}")
            setattr(self, name, value)
        post_init = getattr(self, "__post_init__", None)
        if post_init is not None:
            post_init()

    def __iter__(self):
        return iter(getattr(self, name) for name in self.__class__.__annotations__)


class _ConstantList:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: Any) -> Any:
        return self._values[index]


class _PoolingOutput:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


class _PoolingRequestOutput:
    def __init__(
        self,
        request_id: str,
        outputs: _PoolingOutput,
        prompt_token_ids: list[int],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.outputs = outputs
        self.prompt_token_ids = prompt_token_ids
        self.finished = finished

    def add(self, other: "_PoolingRequestOutput", aggregate: bool) -> None:
        del aggregate
        self.outputs = other.outputs
        self.finished = other.finished


class _RequestOutput:
    pass


class _RequestOutputKind(enum.Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


class _PoolingParams:
    output_kind = _RequestOutputKind.FINAL_ONLY
    prompt_logprobs = None
    truncate_prompt_tokens = None


class _LoRARequestStates:
    def add_request(self, _state: Any) -> None: ...
    def abort_request(self, _state: Any) -> None: ...
    def finish_request(self, _state: Any) -> None: ...
    def update_iteration_stats(self, _stats: Any) -> None: ...
    def get_stats(self, _state: Any) -> None:
        return None


class _Logger:
    def debug(self, *_args: Any, **_kwargs: Any) -> None: ...
    def info(self, *_args: Any, **_kwargs: Any) -> None: ...
    def warning_once(self, *_args: Any, **_kwargs: Any) -> None: ...
    def exception(self, *_args: Any, **_kwargs: Any) -> None: ...


class _EngineGenerateError(RuntimeError):
    pass


class _EngineDeadError(RuntimeError):
    pass


def _length_from_prompt(
    prompt_token_ids: list[int] | None, prompt_embeds: torch.Tensor | None
) -> int:
    return len(prompt_token_ids) if prompt_token_ids is not None else len(prompt_embeds or ())


def _compile_classes(
    path: Path, names: tuple[str, ...], namespace: dict[str, Any]
) -> dict[str, type]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"qualified class extraction mismatch: {path}")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _compile_methods(
    path: Path, class_name: str, names: tuple[str, ...], namespace: dict[str, Any]
) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    if {node.name for node in methods} != set(names):
        raise ValueError(f"qualified method extraction mismatch: {path}:{class_name}")
    holder = ast.ClassDef(
        name="_ExactMethods",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            holder,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: getattr(namespace["_ExactMethods"], name) for name in names}


_AUDITS: dict[str, "BoundaryAudit"] = {}


def _audit_record(boundary: str, request_id: str, **details: Any) -> None:
    audit = _AUDITS.get(str(request_id))
    if audit is not None:
        audit.mark(boundary, **details)
    record_boundary(boundary, str(request_id), **details)


@dataclass
class QualifiedClasses:
    AsyncLLM: type
    EngineCoreAppendRequest: type
    EngineCoreOutput: type
    EngineCoreRequest: type
    EngineCoreOutputs: type
    EngineCoreRequestType: type
    FinishReason: type
    OutputProcessor: type
    Request: type
    RequestOutputCollector: type
    RequestStatus: type
    Scheduler: type
    SchedulerOutput: type
    FCFSRequestQueue: type
    core_methods: dict[str, Any]
    provenance: dict[str, Any]


def load_qualified_classes(source_root: Path, patched_root: Path | None = None) -> QualifiedClasses:
    root = source_root if patched_root is None else patched_root
    msgspec = SimpleNamespace(Struct=_MsgspecStruct)
    engine_ns: dict[str, Any] = {"enum": enum, "msgspec": msgspec, "time": time, "torch": torch}
    engine = _compile_classes(
        source_root / "v1/engine/__init__.py",
        (
            "FinishReason",
            "EngineCoreRequest",
            "EngineCoreAppendRequest",
            "EngineCoreOutput",
            "EngineCoreOutputs",
            "EngineCoreRequestType",
        ),
        engine_ns,
    )
    request_ns: dict[str, Any] = {
        "ConstantList": _ConstantList,
        "EngineCoreEvent": object,
        "EngineCoreEventType": object,
        "FinishReason": engine["FinishReason"],
        "StructuredOutputRequest": object,
        "enum": enum,
        "length_from_prompt_token_ids_or_embeds": _length_from_prompt,
        "partial": partial,
        "threading": __import__("threading"),
        "time": time,
        "torch": torch,
    }
    request = _compile_classes(
        source_root / "v1/request.py", ("Request", "RequestStatus"), request_ns
    )
    output_data_ns: dict[str, Any] = {
        "bc_linter_include": lambda value: value,
        "dataclass": dataclass,
    }
    output_data = _compile_classes(
        source_root / "v1/core/sched/output.py",
        ("NewRequestData", "CachedRequestData", "SchedulerOutput"),
        output_data_ns,
    )
    queue_ns: dict[str, Any] = {
        "ABC": ABC,
        "Iterable": Iterable,
        "Iterator": Iterator,
        "Request": request["Request"],
        "abstractmethod": abstractmethod,
        "deque": deque,
    }
    request_queue = _compile_classes(
        source_root / "v1/core/sched/request_queue.py",
        ("RequestQueue", "FCFSRequestQueue"),
        queue_ns,
    )
    scheduling_policy = SimpleNamespace(FCFS="fcfs", PRIORITY="priority")
    scheduler_ns: dict[str, Any] = {
        "CFG_UNCOND_SUFFIX": ":cfg_uncond",
        "CachedRequestData": output_data["CachedRequestData"],
        "EngineCoreEventType": SimpleNamespace(PREEMPTED=1),
        "KVCacheBlocks": object,
        "KVEventBatch": object,
        "MULTIMODAL_REGISTRY": None,
        "NewRequestData": output_data["NewRequestData"],
        "Request": request["Request"],
        "RequestStatus": request["RequestStatus"],
        "SchedulerInterface": object,
        "SchedulerOutput": output_data["SchedulerOutput"],
        "SchedulingPolicy": scheduling_policy,
        "create_request_queue": lambda _policy: request_queue["FCFSRequestQueue"](),
        "descriptor_key": descriptor_key,
        "itertools": itertools,
        "logger": _Logger(),
        "record_boundary": _audit_record,
        "time": time,
    }
    scheduler = _compile_classes(root / "v1/core/sched/scheduler.py", ("Scheduler",), scheduler_ns)
    output_ns: dict[str, Any] = {
        "CompletionOutput": object,
        "EngineCoreRequest": engine["EngineCoreRequest"],
        "FinishReason": engine["FinishReason"],
        "IncrementalDetokenizer": object,
        "LoRARequestStates": _LoRARequestStates,
        "LogprobsProcessor": object,
        "ParentRequest": SimpleNamespace(observe_finished_request=lambda *_args: None),
        "PoolingOutput": _PoolingOutput,
        "PoolingRequestOutput": _PoolingRequestOutput,
        "RequestOutput": _RequestOutput,
        "RequestOutputKind": _RequestOutputKind,
        "asyncio": asyncio,
        "cast": cast,
        "dataclass": dataclass,
        "length_from_prompt_token_ids_or_embeds": _length_from_prompt,
        "record_boundary": _audit_record,
        "torch": torch,
    }
    output = _compile_classes(
        root / "v1/engine/output_processor.py",
        ("RequestOutputCollector", "OutputProcessorOutput", "RequestState", "OutputProcessor"),
        output_ns,
    )
    async_ns: dict[str, Any] = {
        "EngineClient": object,
        "EngineCoreRequest": engine["EngineCoreRequest"],
        "EngineDeadError": _EngineDeadError,
        "EngineGenerateError": _EngineGenerateError,
        "IterationStats": object,
        "MULTIMODAL_REGISTRY": None,
        "PoolingParams": _PoolingParams,
        "RequestOutputCollector": output["RequestOutputCollector"],
        "UsageContext": SimpleNamespace(ENGINE_CONTEXT=object()),
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE": 4096,
        "_validate_truncation_size": lambda *_args: None,
        "asyncio": asyncio,
        "cdiv": lambda x, y: (x + y - 1) // y,
        "deprecate_kwargs": lambda *_args, **_kwargs: lambda function: function,
        "logger": _Logger(),
        "np": __import__("numpy"),
    }
    async_llm = _compile_classes(source_root / "v1/engine/async_llm.py", ("AsyncLLM",), async_ns)
    core_ns = {
        "EngineCoreOutputs": engine["EngineCoreOutputs"],
        "EngineCoreRequestType": engine["EngineCoreRequestType"],
        "UtilityOutput": object,
        "UtilityResult": object,
        "logger": _Logger(),
        "record_boundary": _audit_record,
    }
    core_methods = _compile_methods(
        root / "v1/engine/core.py",
        "EngineCoreProc",
        ("_handle_client_request", "_process_engine_step"),
        core_ns,
    )
    provenance = {}
    for relative in (
        "v1/engine/__init__.py",
        "v1/request.py",
        "v1/core/sched/request_queue.py",
        "v1/core/sched/output.py",
        "v1/core/sched/scheduler.py",
        "v1/engine/output_processor.py",
        "v1/engine/async_llm.py",
        "v1/engine/core.py",
    ):
        provenance[relative] = {
            "sha256": sha256_file(source_root / relative),
            "execution": "AST-selected exact qualified definition",
        }
    return QualifiedClasses(
        AsyncLLM=async_llm["AsyncLLM"],
        EngineCoreAppendRequest=engine["EngineCoreAppendRequest"],
        EngineCoreOutput=engine["EngineCoreOutput"],
        EngineCoreRequest=engine["EngineCoreRequest"],
        EngineCoreOutputs=engine["EngineCoreOutputs"],
        EngineCoreRequestType=engine["EngineCoreRequestType"],
        FinishReason=engine["FinishReason"],
        OutputProcessor=output["OutputProcessor"],
        Request=request["Request"],
        RequestOutputCollector=output["RequestOutputCollector"],
        RequestStatus=request["RequestStatus"],
        Scheduler=scheduler["Scheduler"],
        SchedulerOutput=output_data["SchedulerOutput"],
        FCFSRequestQueue=request_queue["FCFSRequestQueue"],
        core_methods=core_methods,
        provenance=provenance,
    )


@dataclass
class Decision:
    index: int
    lane_count: int
    append_wait_order: str
    request_order: tuple[int, ...]
    output_order: tuple[int, ...]
    batch_partition: str
    cancellation_target: int
    cancellation_phase: str
    error_target: int
    error_phase: str
    yields: tuple[int, ...]
    replay_vector: tuple[int, ...]

    def expanded(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "request_order": list(self.request_order),
            "output_order": list(self.output_order),
            "yields": list(self.yields),
            "replay_vector": list(self.replay_vector),
            "version": SCHEDULE_VERSION,
        }


def decision_for(index: int) -> Decision:
    words = tuple(splitmix64(ROOT_SEED + index + offset * 0x100000001B3) for offset in range(4))
    lane_count = 1 + int(words[0] & 1)
    request_order = tuple(reversed(range(lane_count))) if words[0] & 2 else tuple(range(lane_count))
    output_order = tuple(reversed(range(lane_count))) if words[0] & 4 else tuple(range(lane_count))
    cancellation_target = int(words[1] % lane_count) if (words[1] >> 8) & 0x3F == 0 else -1
    error_target = (
        int(words[1] % lane_count)
        if cancellation_target < 0 and (words[1] >> 14) & 0x7F == 0
        else -1
    )
    cancellation_phase = "after_B8" if words[1] & 1 else "before_B9"
    error_phase = "after_B8" if words[1] & 2 else "before_B9"
    return Decision(
        index=index,
        lane_count=lane_count,
        append_wait_order="wait_then_append" if words[0] & 8 else "append_then_wait",
        request_order=request_order,
        output_order=output_order,
        batch_partition="together" if words[0] & 16 else "individual",
        cancellation_target=cancellation_target,
        cancellation_phase=cancellation_phase if cancellation_target >= 0 else "inactive",
        error_target=error_target,
        error_phase=error_phase if error_target >= 0 else "inactive",
        yields=tuple((words[2] >> point) & 1 for point in YIELD_POINTS),
        replay_vector=words,
    )


class BoundaryAudit:
    def __init__(self, record_suppression: str | None = None) -> None:
        self.mask = 0
        self.record_suppression = record_suppression
        self.sequence: list[str] = []

    def mark(self, boundary: str, **_details: Any) -> None:
        if boundary == self.record_suppression:
            return
        bit = 1 << int(boundary[1:])
        if not self.mask & bit:
            self.mask |= bit
            self.sequence.append(boundary)

    def first_missing(self) -> str | None:
        return next(
            (boundary for boundary in BOUNDARIES if not self.mask & (1 << int(boundary[1:]))), None
        )


@dataclass
class ScheduleContext:
    decision: Decision
    operation_loss: str | None = None
    record_suppression: str | None = None
    events: list[str] = field(default_factory=list)
    audits: list[BoundaryAudit] = field(default_factory=list)
    added: list[asyncio.Event] = field(default_factory=list)
    appended: list[asyncio.Event] = field(default_factory=list)
    client_add_returned: list[asyncio.Event] = field(default_factory=list)
    realized_yields: list[tuple[int, int]] = field(default_factory=list)

    def event(self, label: str) -> None:
        self.events.append(label)

    async def yield_at(self, point: int, label: str) -> None:
        if point not in YIELD_POINTS:
            raise ValueError(f"unregistered yield point: {point}")
        yielded = self.decision.yields[point]
        self.realized_yields.append((point, yielded))
        self.events.append(f"yield:{point}:{label}|decision={yielded}")
        if yielded:
            await asyncio.sleep(0)


class _Blocks:
    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
        del allow_none
        return ([0],)


class _KVCacheManager:
    def allocate_slots(self, *_args: Any, **_kwargs: Any) -> _Blocks:
        return _Blocks()

    def get_computed_blocks(self, *_args: Any, **_kwargs: Any) -> tuple[_Blocks, int]:
        return _Blocks(), 0

    def get_blocks(self, _request_id: str) -> _Blocks:
        return _Blocks()

    def create_empty_block_list(self) -> _Blocks:
        return _Blocks()

    def get_num_common_prefix_blocks(self, *_args: Any) -> list[int]:
        return [0]

    def take_events(self) -> None:
        return None

    def take_freed_block_ids(self) -> list[int]:
        return []

    def free(self, _request: Any) -> None: ...


class _EncoderCacheManager:
    def get_freed_mm_hashes(self) -> list[str]:
        return []

    def free(self, _request: Any) -> None: ...


def _new_scheduler(qualified: QualifiedClasses, request: Any) -> Any:
    scheduler = qualified.Scheduler.__new__(qualified.Scheduler)
    scheduler.requests = {}
    scheduler.finished_req_ids = set()
    scheduler.await_inputs = True
    scheduler.voicechat_pad_pair_token_id = 0
    scheduler.max_num_scheduled_tokens = 2
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_model_len = 4096
    scheduler.num_lookahead_tokens = 0
    scheduler.waiting_input = set()
    scheduler.running = []
    scheduler.waiting = qualified.FCFSRequestQueue()
    scheduler.cfg_pairs = {}
    scheduler.scheduler_config = SimpleNamespace(
        chunked_prefill_enabled=False,
        long_prefill_token_threshold=0,
        disable_chunked_mm_input=False,
        max_num_encoder_input_tokens=0,
    )
    scheduler.kv_cache_manager = _KVCacheManager()
    scheduler.encoder_cache_manager = _EncoderCacheManager()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.connector = None
    scheduler.kv_event_publisher = SimpleNamespace(publish=lambda _event: None)
    scheduler.policy = "fcfs"
    scheduler.max_num_running_reqs = 64
    scheduler.log_stats = False
    scheduler.lora_config = None
    scheduler.use_pp = False
    scheduler.is_encoder_decoder = False
    scheduler.enable_guidance = False
    scheduler.structured_output_manager = SimpleNamespace()
    scheduler.add_request(request)
    return scheduler


def _new_core_request(qualified: QualifiedClasses, request_id: str, lane: int) -> Any:
    return qualified.EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[lane + 1],
        mm_features=None,
        sampling_params=None,
        pooling_params=_PoolingParams(),
        eos_token_id=None,
        arrival_time=float(lane),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


class _CoreDispatch:
    def __init__(self, runtime: "RaceRuntime") -> None:
        self.runtime = runtime
        self.schedulers: dict[str, Any] = {}
        self.output_queue: asyncio.Queue[Any] = runtime.core_output_queue
        self.step_payload: dict[int, Any] = {}
        self._current_ctx: ScheduleContext | None = None
        self.__class__._handle_client_request = runtime.qualified.core_methods[
            "_handle_client_request"
        ]
        self.__class__._process_engine_step = runtime.qualified.core_methods["_process_engine_step"]

    def add_request(self, request: Any, request_wave: int = 0) -> None:
        del request_wave
        exact = self.runtime.qualified.Request(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=None,
            pooling_params=request.pooling_params,
            eos_token_id=None,
        )
        self.schedulers[request.request_id] = _new_scheduler(self.runtime.qualified, exact)

    def set_custom_inputs(self, request_id: str, custom_inputs: dict[str, torch.Tensor]) -> None:
        ctx = self.runtime.context_for(request_id)
        if ctx.operation_loss == "B2":
            return
        self.schedulers[request_id].set_custom_inputs(request_id, custom_inputs)

    def abort_requests(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.schedulers.pop(request_id, None)

    def step_fn(self) -> tuple[dict[int, Any], bool]:
        assert self._current_ctx is not None
        return self.step_payload.pop(self._current_ctx.decision.index), True

    def post_step(self, _model_executed: bool) -> None: ...


class _EngineClient:
    def __init__(self, runtime: "RaceRuntime") -> None:
        self.runtime = runtime

    async def add_request_async(self, request: Any) -> None:
        ctx = self.runtime.context_for(request.request_id)
        lane = int(request.request_id.rsplit("r", 1)[1])
        await ctx.yield_at(0, "client:before_add")
        await self.runtime.input_queue.put(
            (self.runtime.qualified.EngineCoreRequestType.ADD, (request, 0), ctx)
        )
        await ctx.yield_at(1, "client:after_add")
        ctx.client_add_returned[lane].set()

    async def set_custom_inputs_async(self, request: Any) -> None:
        ctx = self.runtime.context_for(request.request_id)
        await self.runtime.input_queue.put(
            (self.runtime.qualified.EngineCoreRequestType.APPEND, tuple(request), ctx)
        )

    async def get_output_async(self) -> Any:
        # These are true receive-side transition points around the real queue wait.
        active = asyncio.current_task()
        assert active is not None
        ctx = None
        while ctx is None:
            # The output handler is shared, so point 17 belongs to the item that
            # eventually awakens it. Record it after dequeue with points 18/19.
            outputs, ctx = await self.runtime.wire_queue.get()
        await ctx.yield_at(17, "receive:wire_item_acquired")
        await ctx.yield_at(18, "receive:before_output_handler")
        await ctx.yield_at(19, "receive:return_to_output_handler")
        return outputs

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        del request_ids


class _AsyncEngine:
    def __init__(self, runtime: "RaceRuntime") -> None:
        self.runtime = runtime
        self.errored = False
        self.log_requests = False
        self.log_stats = False
        self.logger_manager = None
        self.model_config = SimpleNamespace(max_model_len=4096)
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
        )
        self.output_processor = runtime.qualified.OutputProcessor(tokenizer=None, log_stats=False)
        self.engine_core = _EngineClient(runtime)
        self.output_handler = None
        for name in ("generate", "add_request", "_add_request", "_run_output_handler", "abort"):
            setattr(self.__class__, name, getattr(runtime.qualified.AsyncLLM, name))


class RaceRuntime:
    def __init__(self, qualified: QualifiedClasses) -> None:
        self.qualified = qualified
        self.contexts: dict[int, ScheduleContext] = {}
        self.request_contexts: dict[str, ScheduleContext] = {}
        self.input_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.execute_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.core_output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.wire_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.core = _CoreDispatch(self)
        self.engine = _AsyncEngine(self)
        self.tasks: list[asyncio.Task[Any]] = []

    def context_for(self, request_id: str) -> ScheduleContext:
        return self.request_contexts[request_id]

    async def start(self) -> None:
        self.tasks = [
            asyncio.create_task(self._core_actor(), name="phase0-core-dispatch"),
            asyncio.create_task(self._executor_actor(), name="phase0-fake-executor"),
            asyncio.create_task(self._sender_actor(), name="phase0-core-sender"),
        ]

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        if self.engine.output_handler is not None:
            self.engine.output_handler.cancel()
        await asyncio.gather(*self.tasks, self.engine.output_handler, return_exceptions=True)

    async def _core_actor(self) -> None:
        while True:
            request_type, payload, ctx = await self.input_queue.get()
            await ctx.yield_at(2, "core:after_input_get")
            request_id = (
                payload[0].request_id
                if request_type == self.qualified.EngineCoreRequestType.ADD
                else payload[0]
            )
            lane = int(request_id.rsplit("r", 1)[1])
            if request_type == self.qualified.EngineCoreRequestType.APPEND:
                await ctx.yield_at(6 + lane, f"core:before_append_dispatch:lane={lane}")
            if (
                request_type == self.qualified.EngineCoreRequestType.APPEND
                and ctx.operation_loss == "B1"
            ):
                ctx.appended[lane].set()
                continue
            self.core._handle_client_request(request_type, payload)
            ctx.event(f"core:dispatched:{request_type.name}:lane={lane}")
            if request_type == self.qualified.EngineCoreRequestType.ADD:
                ctx.added[lane].set()
            else:
                ctx.appended[lane].set()
            await ctx.yield_at(3, "core:after_dispatch")

    async def _executor_actor(self) -> None:
        while True:
            request_ids, ctx = await self.execute_queue.get()
            completed_outputs = []
            for request_id in request_ids:
                lane = int(request_id.rsplit("r", 1)[1])
                await ctx.yield_at(8 + lane, "executor:before_schedule")
                scheduler = self.core.schedulers.get(request_id)
                if scheduler is None:
                    continue
                if ctx.operation_loss in {"B0", "B1", "B2"}:
                    ctx.event(
                        f"scheduler:transition_withheld:lane={lane}:loss={ctx.operation_loss}"
                    )
                    continue
                descriptor = scheduler.schedule()
                if request_id not in descriptor.num_scheduled_tokens:
                    continue
                descriptor_value = {
                    "request": request_id,
                    "tokens": descriptor.num_scheduled_tokens[request_id],
                }
                ctx.event(f"scheduler:descriptor:lane={lane}")
                if ctx.operation_loss == "B3":
                    continue
                _audit_record(
                    "B3",
                    request_id,
                    descriptor_key=descriptor_key(descriptor_value),
                    epoch=0,
                    graph_mode="cpu-fake-executor",
                    token_count=descriptor.num_scheduled_tokens[request_id],
                )
                ctx.event(f"executor:entered:lane={lane}")
                await ctx.yield_at(10 + lane, "executor:before_forward")
                if ctx.operation_loss == "B4":
                    continue
                _audit_record("B4", request_id, fake_model_output_returned=True)
                ctx.event(f"executor:forward_returned:lane={lane}")
                if ctx.operation_loss == "B5":
                    continue
                _audit_record(
                    "B5",
                    request_id,
                    accepted_count=1,
                    commit_outcome="fake_executor_committed",
                    proposed_count=1,
                )
                ctx.event(f"executor:sample_published:lane={lane}")
                completed_outputs.append(
                    self.qualified.EngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=[],
                        pooling_output=torch.tensor([ctx.decision.index, lane], dtype=torch.int64),
                        finish_reason=(
                            None
                            if ctx.operation_loss == "B9"
                            else self.qualified.FinishReason.STOP
                        ),
                    )
                )
            if ctx.operation_loss == "B6":
                continue
            chunks = (
                [completed_outputs]
                if ctx.decision.batch_partition == "together"
                else [[output] for output in completed_outputs]
            )
            for chunk in chunks:
                if not chunk:
                    continue
                outputs = self.qualified.EngineCoreOutputs(outputs=chunk)
                ctx.event(
                    "core_output:chunk:lanes="
                    + ",".join(item.request_id.rsplit("r", 1)[1] for item in chunk)
                )
                self.core.step_payload[ctx.decision.index] = {0: outputs}
                self.core._current_ctx = ctx
                await ctx.yield_at(12, "executor:before_core_enqueue")
                self.core._process_engine_step()
                await ctx.yield_at(13, "executor:after_core_enqueue")

    async def _sender_actor(self) -> None:
        while True:
            _client_index, outputs = await self.core_output_queue.get()
            request_id = outputs.outputs[0].request_id
            ctx = self.context_for(request_id)
            await ctx.yield_at(14, "sender:after_core_get")
            if ctx.operation_loss == "B7":
                continue
            await ctx.yield_at(15, "sender:before_wire_put")
            await self.wire_queue.put((outputs, ctx))
            for item in outputs.outputs:
                _audit_record("B7", item.request_id, queue_interface="asyncio.Queue", sent=True)
            ctx.event(f"sender:wire_put:size={len(outputs.outputs)}")
            await ctx.yield_at(16, "sender:after_wire_put")


async def _consume(
    runtime: RaceRuntime, request: Any, request_id: str, ctx: ScheduleContext, lane: int
) -> dict[str, Any]:
    iterator = runtime.engine.generate(request, _PoolingParams(), request_id)
    try:
        output = await iterator.__anext__()
        ctx.event(f"iterator:first_output:lane={lane}:finished={output.finished}")
        await ctx.yield_at(20 + lane, f"iterator:awakened:lane={lane}")
        if ctx.operation_loss == "B9":
            await ctx.yield_at(22 + lane, f"iterator:before_stranded_wait:lane={lane}")
            ctx.event(f"iterator:stranded_wait:lane={lane}")
            # B0--B8 completed on the real non-final publication above. No
            # second output is published, so the exact generator and this host
            # consumer remain blocked until fault-matrix cleanup cancels them.
            await iterator.__anext__()
            raise AssertionError("B9 operation-loss iterator unexpectedly returned")
        payload = output.outputs.data.tolist()
        if lane == ctx.decision.cancellation_target:
            if ctx.decision.cancellation_phase == "before_B9":
                await ctx.yield_at(22 + lane, f"host:before_B9:lane={lane}")
            await iterator.aclose()
            ctx.event(
                f"host:cancellation:lane={lane}:phase={ctx.decision.cancellation_phase}"
            )
            if ctx.decision.cancellation_phase == "after_B8":
                await ctx.yield_at(22 + lane, f"host:before_B9:lane={lane}")
            _audit_record(
                "B9",
                request_id,
                cancellation_kind=ctx.decision.cancellation_phase,
                host_call_returned=True,
                iterator_awakened=True,
            )
            return {
                "kind": "cancelled",
                "payload": payload,
                "request_id": request_id,
            }
        if lane == ctx.decision.error_target:
            if ctx.decision.error_phase == "before_B9":
                await ctx.yield_at(22 + lane, f"host:before_B9:lane={lane}")
            error_task = asyncio.create_task(
                _task_error(ctx.decision.index),
                name=f"phase0-task-error-{ctx.decision.index}-{lane}",
            )
            try:
                await error_task
            except RuntimeError:
                ctx.event(f"host:error:lane={lane}:phase={ctx.decision.error_phase}")
                if ctx.decision.error_phase == "after_B8":
                    await ctx.yield_at(22 + lane, f"host:before_B9:lane={lane}")
                _audit_record(
                    "B9",
                    request_id,
                    exception_kind="RuntimeError",
                    host_call_returned=True,
                    iterator_awakened=True,
                )
                return {
                    "kind": "task_error",
                    "payload": payload,
                    "request_id": request_id,
                }
        await ctx.yield_at(22 + lane, f"host:before_B9:lane={lane}")
        ctx.event(f"host:return:lane={lane}")
        _audit_record("B9", request_id, host_call_returned=True, iterator_awakened=True)
        return {"kind": "output", "payload": payload, "request_id": output.request_id}
    except asyncio.CancelledError:
        await iterator.aclose()
        return {"kind": "cleanup_cancelled", "request_id": request_id}
    except _EngineGenerateError:
        _audit_record(
            "B9",
            request_id,
            exception_kind="EngineGenerateError",
            host_call_returned=True,
            iterator_awakened=True,
        )
        return {"kind": "task_error", "request_id": request_id}


async def _task_error(schedule_index: int) -> None:
    await asyncio.sleep(0)
    raise RuntimeError(f"seeded task error for schedule {schedule_index}")


async def run_schedule_async(
    runtime: RaceRuntime,
    schedule_index: int,
    *,
    operation_loss: str | None = None,
    record_suppression: str | None = None,
) -> dict[str, Any]:
    decision = decision_for(schedule_index)
    ctx = ScheduleContext(decision, operation_loss, record_suppression)
    ctx.audits = [BoundaryAudit(record_suppression) for _ in range(decision.lane_count)]
    ctx.added = [asyncio.Event() for _ in range(decision.lane_count)]
    ctx.appended = [asyncio.Event() for _ in range(decision.lane_count)]
    ctx.client_add_returned = [asyncio.Event() for _ in range(decision.lane_count)]
    runtime.contexts[schedule_index] = ctx
    consumers: list[asyncio.Task[dict[str, Any]]] = []
    request_ids = [f"v2s{schedule_index:x}-r{lane}" for lane in range(decision.lane_count)]
    for lane, request_id in enumerate(request_ids):
        _AUDITS[request_id] = ctx.audits[lane]
        runtime.request_contexts[request_id] = ctx
        request = _new_core_request(runtime.qualified, request_id, lane)
        consumers.append(
            asyncio.create_task(
                _consume(runtime, request, request_id, ctx, lane),
                name=f"phase0-generate-{schedule_index}-{lane}",
            )
        )
    for lane in decision.request_order:
        await ctx.added[lane].wait()
        await ctx.client_add_returned[lane].wait()
    if decision.append_wait_order == "wait_then_append":
        # Let the exact generate task enter RequestOutputCollector.get() before
        # the append. Scheduler admission itself remains in its exact WAITING
        # queue until the append makes the request runnable.
        await asyncio.sleep(0)
        ctx.event("topology:wait_then_append")
    else:
        ctx.event("topology:append_then_wait")
    for lane in decision.request_order:
        request_id = request_ids[lane]
        await ctx.yield_at(
            4 + lane,
            f"host:{decision.append_wait_order}:before_append:lane={lane}",
        )
        if operation_loss == "B0":
            ctx.appended[lane].set()
            continue
        _audit_record("B0", request_id, append_intent=True, host_packed_call=True)
        append = runtime.qualified.EngineCoreAppendRequest(
            request_id, {"combined_embeds": torch.zeros((2, 1))}
        )
        await runtime.engine.engine_core.set_custom_inputs_async(append)
    for lane in decision.request_order:
        await ctx.appended[lane].wait()
    await runtime.execute_queue.put(([request_ids[lane] for lane in decision.output_order], ctx))
    if operation_loss == "B8":
        for request_id in request_ids:
            runtime.engine.output_processor.request_states.pop(request_id, None)
    stranded = operation_loss is not None
    pending_before_cleanup: list[str] = []
    if stranded:
        done, pending = await asyncio.wait(consumers, timeout=0.002)
        pending_before_cleanup = sorted(task.get_name() for task in pending)
        for task in pending:
            task.cancel()
        routed = list(await asyncio.gather(*consumers, return_exceptions=True))
    else:
        routed = list(await asyncio.gather(*consumers))
    routing_mismatches = 0
    for lane, item in enumerate(routed):
        if not isinstance(item, dict):
            continue
        if item.get("kind") == "output" and item.get("payload") != [schedule_index, lane]:
            routing_mismatches += 1
        if item.get("request_id") != request_ids[lane]:
            routing_mismatches += 1
    normalized_events = list(ctx.events)
    equivalence_class = stable_digest(normalized_events)[:20]
    first_missing = [audit.first_missing() for audit in ctx.audits]
    for request_id in request_ids:
        _AUDITS.pop(request_id, None)
        runtime.request_contexts.pop(request_id, None)
    runtime.contexts.pop(schedule_index, None)
    return {
        "decision": decision.expanded(),
        "equivalence_class": equivalence_class,
        "first_missing": first_missing,
        "lane_count": decision.lane_count,
        "pending_before_cleanup": pending_before_cleanup,
        "realized_events": normalized_events,
        "realized_yields": [list(item) for item in ctx.realized_yields],
        "routing_digest": stable_digest(routed),
        "routing_mismatches": routing_mismatches,
    }


def compatibility_identity(
    patch_manifest: dict[str, Any] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    files = {relative: sha256_file(REPO_ROOT / relative) for relative in HARNESS_FILES}
    identity: dict[str, Any] = {
        "application_revision": verify_application_revision(),
        "configuration": {
            "batch_size": batch_size,
            "concurrency": concurrency,
            "distinct_interleavings": REQUIRED_DISTINCT_INTERLEAVINGS,
            "root_seed": ROOT_SEED,
            "schedule_version": SCHEDULE_VERSION,
            "schedules": REQUIRED_SCHEDULES,
            "yield_points": list(YIELD_POINTS),
        },
        "design_sha256": sha256_file(DESIGN_PATH),
        "frozen_harness": files,
        "preregistration_sha256": sha256_file(PREREG_PATH),
        "qualified_tree_sha256": QUALIFIED_TREE_SHA256,
        "runner_sha256": files["tools/qualification/step5_phase0_protocol_race.py"],
    }
    if patch_manifest is not None:
        identity["patch_manifest_sha256"] = stable_digest(patch_manifest)
    identity["identity_sha256"] = stable_digest(identity)
    return identity


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


ARTIFACT_MANIFEST_NAME = "artifact-manifest-v3.json"


def verify_artifact_manifest(output_dir: Path) -> bool:
    manifest_path = output_dir / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(manifest["files"]) != expected_paths:
        return False
    for relative, expected in manifest["files"].items():
        path = output_dir / relative
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            return False
    canonical = dict(manifest)
    claimed_self = canonical.pop("self_canonical_sha256")
    return claimed_self == stable_digest(canonical)


def write_and_verify_artifact_manifest(output_dir: Path, campaign_passed: bool) -> bool:
    manifest_path = output_dir / ARTIFACT_MANIFEST_NAME
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            files[str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest: dict[str, Any] = {
        "campaign_passed": campaign_passed,
        "files": files,
        "schema_version": 3,
        "self_hash_definition": (
            "SHA-256 of canonical manifest JSON with self_canonical_sha256 omitted"
        ),
    }
    manifest["self_canonical_sha256"] = stable_digest(manifest)
    atomic_json(manifest_path, manifest)
    return verify_artifact_manifest(output_dir)


def _histogram_update(histograms: dict[str, Counter[str]], result: dict[str, Any]) -> None:
    decision = result["decision"]
    for key in ("append_wait_order", "batch_partition", "cancellation_phase", "error_phase"):
        histograms[key][str(decision[key])] += 1
    histograms["request_order"][str(decision["request_order"])] += 1
    histograms["output_order"][str(decision["output_order"])] += 1
    histograms["cancellation_target"][str(decision["cancellation_target"])] += 1
    histograms["error_target"][str(decision["error_target"])] += 1
    for point, value in result["realized_yields"]:
        histograms[f"yield_{point}"][str(value)] += 1


async def _run_range(
    runtime: RaceRuntime, start: int, end: int, concurrency: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for chunk_start in range(start, end, concurrency):
        chunk_end = min(chunk_start + concurrency, end)
        results.extend(
            await asyncio.gather(
                *(run_schedule_async(runtime, index) for index in range(chunk_start, chunk_end))
            )
        )
    return results


def _validate_resume(
    batch_path: Path, identity: dict[str, Any]
) -> tuple[int, Counter[str], Counter[str], dict[str, dict[str, Any]]]:
    expected_start = 0
    totals: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    representatives: dict[str, dict[str, Any]] = {}
    if not batch_path.exists():
        return expected_start, totals, classes, representatives
    with batch_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            batch = json.loads(line)
            if (
                batch["start_schedule"] != expected_start
                or batch["end_schedule_exclusive"] <= expected_start
            ):
                raise ValueError(f"resume range gap/overlap at line {line_number}")
            if batch["compatibility_identity_sha256"] != identity["identity_sha256"]:
                raise ValueError(f"resume compatibility mismatch at line {line_number}")
            if batch["batch_sha256"] != stable_digest(
                {key: value for key, value in batch.items() if key != "batch_sha256"}
            ):
                raise ValueError(f"resume batch hash mismatch at line {line_number}")
            expected_start = batch["end_schedule_exclusive"]
            totals.update(batch["counts"])
            classes.update(batch["equivalence_classes"])
            for class_id, local in batch["representatives"].items():
                item = representatives.setdefault(
                    class_id,
                    {
                        "count": 0,
                        "first_index": local["first_index"],
                        "last_index": local["last_index"],
                        "expanded_schedule": local["expanded_schedule"],
                        "realized_events": local["realized_events"],
                    },
                )
                item["count"] += local["count"]
                item["last_index"] = local["last_index"]
    if set(representatives) != set(classes):
        raise ValueError("resume representative/class coverage mismatch")
    for class_id, count in classes.items():
        if representatives[class_id]["count"] != count:
            raise ValueError(f"resume representative count mismatch: {class_id}")
    return expected_start, totals, classes, representatives


async def run_fault_matrices(runtime: RaceRuntime) -> dict[str, Any]:
    operation = []
    suppression = []
    for boundary in BOUNDARIES:
        seed = domain_seed(f"fault/{boundary}")
        index = seed % REQUIRED_SCHEDULES
        op = await run_schedule_async(runtime, index, operation_loss=boundary)
        rec = await run_schedule_async(runtime, index, record_suppression=boundary)
        operation.append(
            {
                "detected": (
                    all(value == boundary for value in op["first_missing"])
                    and (
                        boundary != "B9"
                        or len(op["pending_before_cleanup"]) == op["lane_count"]
                    )
                ),
                "domain_seed": seed,
                "first_missing": op["first_missing"],
                "injected_boundary": boundary,
                "pending_before_cleanup": op["pending_before_cleanup"],
                "schedule": index,
            }
        )
        suppression.append(
            {
                "detected": all(value == boundary for value in rec["first_missing"]),
                "domain_seed": seed,
                "first_missing": rec["first_missing"],
                "injected_boundary": boundary,
                "schedule": index,
            }
        )
    return {
        "operation_loss": operation,
        "operation_loss_detected": sum(item["detected"] for item in operation),
        "record_suppression": suppression,
        "record_suppression_detected": sum(item["detected"] for item in suppression),
    }


def run_noninterference(qualified_root: Path, output_dir: Path) -> dict[str, Any]:
    results = {}
    trace_dir = output_dir / "noninterference-trace"
    for mode in ("off", "on"):
        env = dict(os.environ)
        env.update(
            {
                "NEMOTRON_WEDGE_TRACE": "1" if mode == "on" else "0",
                "NEMOTRON_WEDGE_TRACE_CAPACITY": "4096",
                "NEMOTRON_WEDGE_TRACE_DIR": str(trace_dir),
                "NEMOTRON_WEDGE_TRACE_PROCESS_ROLE": "phase0-noninterference",
                "NEMOTRON_WEDGE_TRACE_RUN_UUID": f"phase0-v3-{mode}",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "fixture",
                "--qualified-vllm-root",
                str(qualified_root),
                "--schedules",
                "64",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        results[mode] = json.loads(completed.stdout)
    cost_env = dict(os.environ)
    cost_env["NEMOTRON_WEDGE_TRACE"] = "0"
    cost = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "cost"],
        check=True,
        capture_output=True,
        text=True,
        env=cost_env,
    )
    retained_ring = Path(results["on"]["trace"]["path"])
    retained_data = retained_ring.read_bytes()
    retained_records = sum(
        bool(retained_data[offset : offset + 512].rstrip(b"\0"))
        for offset in range(0, len(retained_data), 512)
    )
    ring_truthful = retained_records == results["on"]["trace"]["records"]
    return {
        "exact_outputs": results["off"]["fixture"] == results["on"]["fixture"],
        "off": results["off"],
        "on": results["on"],
        "passed": results["off"]["fixture"] == results["on"]["fixture"]
        and results["on"]["trace"]["dropped"] == 0
        and ring_truthful,
        "retained_ring_records": retained_records,
        "retained_ring_truthful": ring_truthful,
        "trace_off_cost": json.loads(cost.stdout),
        "residual": (
            "GPU worker, CUDA graph, Mamba device state, device tensors, and "
            "real ZMQ process timing remain untested."
        ),
    }


async def _fixture_async(qualified_root: Path, schedules: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="step5-v2-fixture-") as temporary:
        patched = Path(temporary) / "patched"
        apply_patch(qualified_root, patched)
        qualified = load_qualified_classes(qualified_root, patched)
        runtime = RaceRuntime(qualified)
        await runtime.start()
        try:
            results = await _run_range(runtime, 0, schedules, 32)
            packed_host = await _packed_host_fixture()
        finally:
            await runtime.close()
    return {
        "missing": sum(
            value is not None for result in results for value in result["first_missing"]
        ),
        "output_sha256": stable_digest(
            {
                "packed_host": packed_host,
                "protocol": [
                    {
                        key: result[key]
                        for key in (
                            "equivalence_class",
                            "first_missing",
                            "routing_digest",
                            "routing_mismatches",
                        )
                    }
                    for result in results
                ],
            }
        ),
        "production_packed_host": packed_host,
        "routing_mismatches": sum(result["routing_mismatches"] for result in results),
        "schedules": schedules,
    }


async def _packed_host_fixture() -> dict[str, Any]:
    """Execute the instrumented production packed host function without GPU code."""
    module_name = "nemo.collections.speechlm2.inference.vllm.streaming_llm_engine"
    parent_names = (
        "nemo",
        "nemo.collections",
        "nemo.collections.speechlm2",
        "nemo.collections.speechlm2.inference",
        "nemo.collections.speechlm2.inference.vllm",
    )

    class GenerationResult:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class StreamStatus:
        FINISHED = "finished"

    fake_module = types.ModuleType(module_name)
    fake_module.GenerationResult = GenerationResult
    fake_module.StreamStatus = StreamStatus
    saved = {name: sys.modules.get(name) for name in (*parent_names, module_name)}
    for name in parent_names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[module_name] = fake_module

    class Iterator:
        def __aiter__(self) -> "Iterator":
            return self

        async def __anext__(self) -> Any:
            return SimpleNamespace(
                finished=False,
                outputs=[
                    SimpleNamespace(
                        custom_outputs=None,
                        finish_reason=None,
                        token_ids=[11, 12],
                    )
                ],
            )

    class AppendClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def append_request(self, **values: Any) -> None:
            self.calls.append(values)

    append_client = AppendClient()
    request_state = SimpleNamespace(
        generated_tokens=[11],
        generation_iterator=Iterator(),
        status=None,
    )
    engine = SimpleNamespace(
        custom_input_specs=[{"dtype": "float32", "name": "combined_embeds"}],
        engine=append_client,
        requests={"frontend": request_state},
        resolve_backend_request_id=lambda _request_id: "backend",
    )
    try:
        from nemotron_voicechat_runtime.runtime_optimizations import (
            _generate_packed_pad_pair,
        )

        result, accepted = await _generate_packed_pad_pair(engine, torch.zeros((2, 1)), "frontend")
        return {
            "accepted": accepted,
            "append_calls": len(append_client.calls),
            "backend_request_id": append_client.calls[0]["request_id"],
            "token_id": result.token_id,
            "total_tokens": result.total_tokens,
        }
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def trace_off_cost() -> dict[str, Any]:
    from nemotron_voicechat_runtime.wedge_boundary_trace import record_boundary as trace_call

    iterations = 20_000
    started = time.perf_counter_ns()
    for index in range(iterations):
        trace_call("B0", "cost", index=index, append_intent=True)
    elapsed = time.perf_counter_ns() - started
    return {
        "allocations": (
            "one kwargs dict plus argument scalar construction at each instrumented call site"
        ),
        "branches": "one _RING-is-None branch inside record_boundary",
        "measured_nanoseconds_per_disabled_call": elapsed / iterations,
    }


async def run_campaign_async(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_identity = verify_qualified_tree(args.qualified_vllm_root.resolve())
    with tempfile.TemporaryDirectory(prefix="step5-v2-patched-") as temporary:
        patched = Path(temporary) / "patched"
        patch_manifest = apply_patch(args.qualified_vllm_root.resolve(), patched)
        prereg_patch = json.loads(
            (output_dir / "diagnostic-patch-manifest-preregistered.json").read_text()
        )
        if patch_manifest != prereg_patch:
            raise ValueError("diagnostic patch differs from preregistered manifest")
        identity = compatibility_identity(
            patch_manifest,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )
        prereg_identity = json.loads(
            (output_dir / "compatibility-identity-preregistered.json").read_text()
        )
        if identity != prereg_identity:
            raise ValueError("campaign compatibility identity differs from preregistration")
        qualified = load_qualified_classes(args.qualified_vllm_root.resolve(), patched)
        atomic_json(output_dir / "source-identity.json", source_identity)
        atomic_json(output_dir / "class-provenance.json", qualified.provenance)
        batch_path = output_dir / "batches-v3.jsonl"
        if batch_path.exists() and not args.resume:
            raise FileExistsError(f"refusing to overwrite {batch_path}")
        start_index, totals, class_counts, representatives = (
            _validate_resume(batch_path, identity)
            if args.resume
            else (0, Counter(), Counter(), {})
        )
        runtime = RaceRuntime(qualified)
        await runtime.start()
        started = time.monotonic()
        try:
            with batch_path.open("a", encoding="utf-8") as stream:
                for batch_start in range(start_index, args.schedules, args.batch_size):
                    batch_end = min(batch_start + args.batch_size, args.schedules)
                    batch_started = time.monotonic()
                    results = await _run_range(runtime, batch_start, batch_end, args.concurrency)
                    counts: Counter[str] = Counter()
                    classes: Counter[str] = Counter()
                    batch_representatives: dict[str, dict[str, Any]] = {}
                    histograms: dict[str, Counter[str]] = {}
                    for result in results:
                        counts["schedules"] += 1
                        counts["requests"] += result["lane_count"]
                        counts["missing_completions"] += sum(
                            value is not None for value in result["first_missing"]
                        )
                        counts["routing_mismatches"] += result["routing_mismatches"]
                        classes[result["equivalence_class"]] += 1
                        hist_default = {
                            key: Counter()
                            for key in (
                                "append_wait_order",
                                "batch_partition",
                                "cancellation_phase",
                                "error_phase",
                                "request_order",
                                "output_order",
                                "cancellation_target",
                                "error_target",
                                *(f"yield_{point}" for point in YIELD_POINTS),
                            )
                        }
                        for key, value in hist_default.items():
                            histograms.setdefault(key, value)
                        _histogram_update(histograms, result)
                        item = batch_representatives.setdefault(
                            result["equivalence_class"],
                            {
                                "count": 0,
                                "first_index": result["decision"]["index"],
                                "last_index": result["decision"]["index"],
                                "expanded_schedule": result["decision"],
                                "realized_events": result["realized_events"],
                            },
                        )
                        item["count"] += 1
                        item["last_index"] = result["decision"]["index"]
                    for class_id, local in batch_representatives.items():
                        item = representatives.setdefault(
                            class_id,
                            {
                                "count": 0,
                                "first_index": local["first_index"],
                                "last_index": local["last_index"],
                                "expanded_schedule": local["expanded_schedule"],
                                "realized_events": local["realized_events"],
                            },
                        )
                        item["count"] += local["count"]
                        item["last_index"] = local["last_index"]
                    batch: dict[str, Any] = {
                        "compatibility_identity_sha256": identity["identity_sha256"],
                        "counts": dict(counts),
                        "elapsed_seconds": time.monotonic() - batch_started,
                        "end_schedule_exclusive": batch_end,
                        "equivalence_classes": dict(classes),
                        "histograms": {key: dict(value) for key, value in histograms.items()},
                        "representatives": batch_representatives,
                        "start_schedule": batch_start,
                    }
                    batch["batch_sha256"] = stable_digest(batch)
                    stream.write(json.dumps(batch, sort_keys=True) + "\n")
                    stream.flush()
                    totals.update(counts)
                    class_counts.update(classes)
            faults = await run_fault_matrices(runtime)
        finally:
            await runtime.close()
    atomic_json(
        output_dir / "equivalence-summary-v3.json",
        {
            "distinct": len(class_counts),
            "representative_count": len(representatives),
            "representatives_retained_in": "batches-v3.jsonl",
            "total_class_memberships": sum(class_counts.values()),
        },
    )
    atomic_json(output_dir / "fault-matrices-v3.json", faults)
    noninterference = run_noninterference(args.qualified_vllm_root.resolve(), output_dir)
    atomic_json(output_dir / "noninterference-v3.json", noninterference)
    nonartifact_passed = (
        identity == prereg_identity
        and totals["schedules"] == REQUIRED_SCHEDULES
        and totals["missing_completions"] == 0
        and totals["routing_mismatches"] == 0
        and len(class_counts) >= REQUIRED_DISTINCT_INTERLEAVINGS
        and len(representatives) == len(class_counts)
        and faults["operation_loss_detected"] == 10
        and faults["record_suppression_detected"] == 10
        and noninterference["passed"]
    )
    summary = {
        "artifact_gate_required": True,
        "compatibility_identity_sha256": identity["identity_sha256"],
        "distinct_interleavings": len(class_counts),
        "elapsed_seconds": time.monotonic() - started,
        "fault_operation_loss_detected": faults["operation_loss_detected"],
        "fault_record_suppression_detected": faults["record_suppression_detected"],
        "noninterference_passed": noninterference["passed"],
        "nonartifact_predicates_passed": nonartifact_passed,
        "passed": nonartifact_passed,
        "preregistered_distinct_interleavings": REQUIRED_DISTINCT_INTERLEAVINGS,
        "totals": dict(totals),
    }
    atomic_json(output_dir / "phase0-summary-v3.json", summary)
    artifact_verified = write_and_verify_artifact_manifest(output_dir, nonartifact_passed)
    summary["artifact_hashes_verified"] = artifact_verified
    summary["passed"] = nonartifact_passed and artifact_verified
    # Rewriting summary changes its payload hash, so rebuild and verify the
    # manifest once with the final PASS fields.
    atomic_json(output_dir / "phase0-summary-v3.json", summary)
    artifact_verified = write_and_verify_artifact_manifest(output_dir, summary["passed"])
    if not artifact_verified:
        summary["artifact_hashes_verified"] = False
        summary["passed"] = False
        atomic_json(output_dir / "phase0-summary-v3.json", summary)
        write_and_verify_artifact_manifest(output_dir, False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--qualified-vllm-root", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--schedules", type=int, default=REQUIRED_SCHEDULES)
    run_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run_parser.add_argument("--resume", action="store_true")
    fixture_parser = subparsers.add_parser("fixture")
    fixture_parser.add_argument("--qualified-vllm-root", type=Path, required=True)
    fixture_parser.add_argument("--schedules", type=int, default=64)
    subparsers.add_parser("cost")
    args = parser.parse_args()
    if args.command == "cost":
        print(json.dumps(trace_off_cost(), sort_keys=True))
        return
    if args.command == "fixture":
        fixture = asyncio.run(_fixture_async(args.qualified_vllm_root.resolve(), args.schedules))
        result = {"fixture": fixture, "trace": trace_metadata()}
        close_trace()
        print(json.dumps(result, sort_keys=True))
        return
    if args.schedules != REQUIRED_SCHEDULES:
        raise SystemExit(f"v2 campaign requires exactly {REQUIRED_SCHEDULES} schedules")
    print(json.dumps(asyncio.run(run_campaign_async(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
