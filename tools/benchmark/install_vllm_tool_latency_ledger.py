#!/usr/bin/env python3
"""Install the opt-in vLLM EngineCore latency ledger in a diagnostic image."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

EXPECTED_CORE_SHA256 = "590925b3823b0bcb4153329441e8e83b1a3ca06b4746082956bdb8afd4004f29"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError("diagnostic vLLM patch context is not unique")
    return source.replace(before, after, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vLLM package is unavailable")
    core = Path(next(iter(spec.submodule_search_locations))) / "v1/engine/core.py"
    original = core.read_bytes()
    if sha256(original) != EXPECTED_CORE_SHA256:
        raise RuntimeError("reconstructed vLLM EngineCore differs from the qualified input")
    source = original.decode("utf-8")
    source = replace_once(source, "import gc\nimport os\n", "import gc\nimport json\nimport os\n")
    source = replace_once(
        source,
        "        self.log_stats = log_stats\n\n        # Setup Model.\n",
        """        self.log_stats = log_stats
        benchmark_value = os.environ.get("S2S_FC_ASYNC_BENCHMARK", "0").strip().lower()
        if benchmark_value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("S2S_FC_ASYNC_BENCHMARK must be a boolean")
        self._voicechat_benchmark_enabled = benchmark_value in {"1", "true", "yes", "on"}
        self._voicechat_benchmark_iteration = 0

        # Setup Model.
""",
    )
    original_step = """        scheduler_output = self.scheduler.schedule()
        model_output = self.execute_model_with_error_logging(
            self.model_executor.execute_model,  # type: ignore
            scheduler_output,
        )
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )  # type: ignore

        return (engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0)
"""
    instrumented_step = """        if not self._voicechat_benchmark_enabled:
            scheduler_output = self.scheduler.schedule()
            model_output = self.execute_model_with_error_logging(
                self.model_executor.execute_model,  # type: ignore
                scheduler_output,
            )
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output
            )  # type: ignore
            return (engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0)

        total_start = time.perf_counter()
        schedule_start = total_start
        scheduler_output = self.scheduler.schedule()
        schedule_end = time.perf_counter()
        execute_start = schedule_end
        model_output = self.execute_model_with_error_logging(
            self.model_executor.execute_model,  # type: ignore
            scheduler_output,
        )
        execute_end = time.perf_counter()
        update_start = execute_end
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )  # type: ignore
        update_end = time.perf_counter()
        schedule_ms = (schedule_end - schedule_start) * 1000.0
        execute_ms = (execute_end - execute_start) * 1000.0
        update_ms = (update_end - update_start) * 1000.0
        total_ms = (update_end - total_start) * 1000.0
        named_sum_ms = schedule_ms + execute_ms + update_ms
        record = {
            "schema": 1,
            "iteration": self._voicechat_benchmark_iteration,
            "request_ids": sorted(scheduler_output.num_scheduled_tokens),
            "scheduled_tokens": {
                str(key): int(value)
                for key, value in scheduler_output.num_scheduled_tokens.items()
            },
            "schedule_ms": schedule_ms,
            "model_executor_ms": execute_ms,
            "scheduler_update_ms": update_ms,
            "named_sum_ms": named_sum_ms,
            "total_ms": total_ms,
            "residual_ms": total_ms - named_sum_ms,
            "monotonic_s": update_end,
        }
        encoded_record = json.dumps(record, sort_keys=True, separators=(",", ":"))
        logger.warning("[VOICECHAT_ENGINE_STEP] %s", encoded_record)
        self._voicechat_benchmark_iteration += 1
        return (engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0)
"""
    source = replace_once(source, original_step, instrumented_step)
    core.write_text(source, encoding="utf-8")
    print(f"installed EngineCore diagnostic ledger: {sha256(core.read_bytes())}")


if __name__ == "__main__":
    main()
