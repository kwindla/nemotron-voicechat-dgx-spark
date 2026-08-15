from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

QUALIFICATION = Path(__file__).parents[2] / "tools" / "qualification"
sys.path.insert(0, str(QUALIFICATION))
MODULE_PATH = QUALIFICATION / "step4_pipecat_soak.py"
SPEC = importlib.util.spec_from_file_location("step4_pipecat_soak", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
soak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(soak)


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_health_check_requires_free_single_client_lock(monkeypatch) -> None:
    monkeypatch.setattr(
        soak.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"status":"ready","active_client":false}'),
    )
    assert soak.read_health("http://127.0.0.1:8786/health")["active_client"] is False

    monkeypatch.setattr(
        soak.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"status":"ready","active_client":true}'),
    )
    with pytest.raises(RuntimeError, match="single-client lock"):
        soak.read_health("http://127.0.0.1:8786/health")


def test_ab_prompt_sequence_is_fixed_and_tool_free() -> None:
    assert len(soak.PROMPTS) == 4
    assert all("tool" not in prompt.lower() for prompt in soak.PROMPTS)


def test_step4c_fixture_is_one_exact_long_l1_after_fixed_warmup() -> None:
    assert soak.QUALIFICATION_FIXTURE == "step4c-l1"
    assert soak.STEP4C_WARMUP_SECONDS == 10.0
    assert soak.STEP4C_DURATION_SECONDS == 120.0
    assert soak.STEP4C_DRAIN_SECONDS == 15.0
    assert soak.L1_TEXT.startswith("The quick brown fox")
    assert soak.L1_TEXT.endswith("northern gate.")


def test_step4c_runner_rejects_missing_playout_path_before_health(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("NEMOTRON_VOICECHAT_PLAYOUT_TRACE", raising=False)
    monkeypatch.setattr(
        soak,
        "read_health",
        lambda _url: pytest.fail("configuration must fail before a live health query"),
    )
    args = SimpleNamespace(
        qualification_fixture=soak.QUALIFICATION_FIXTURE,
        warmup_seconds=10.0,
        duration_seconds=120.0,
        health_url="http://unused",
        output=tmp_path / "unused",
    )

    with pytest.raises(RuntimeError, match="requires a configured.*playout trace"):
        import asyncio

        asyncio.run(soak.run(args))


def _job(ordinal: int, *, status: str = "completed", response_id: str | None = None):
    return {
        "prompt_ordinal": ordinal,
        "prompt_id": f"prompt-{ordinal:02d}",
        "prompt_text": f"prompt {ordinal}",
        "job_id": f"job-{ordinal}",
        "response_id": response_id or f"response-{ordinal}",
        "terminal_status": status,
    }


def test_ab_pass_gate_requires_exactly_one_terminal_per_job() -> None:
    assert soak.jobs_pass([_job(1), _job(2)], 2, []) is True
    assert soak.jobs_pass([_job(1)], 2, []) is False
    assert soak.jobs_pass([_job(1), _job(2, status="submitted")], 2, []) is False
    assert soak.jobs_pass([_job(1), _job(2, response_id="response-1")], 2, []) is False
    assert soak.jobs_pass([_job(1), _job(2)], 2, ["orphan terminal"]) is False


def test_ab_pass_gate_rejects_missing_middle_ordinal() -> None:
    assert soak.jobs_pass([_job(1), _job(3)], 2, []) is False


def test_traced_terminal_identity_is_backfilled_from_closed_artifact(tmp_path) -> None:
    records = [_job(1, status="awaiting_trace", response_id="placeholder")]
    records[0]["response_id"] = None
    trace = tmp_path / "playout.jsonl"
    trace.write_text(
        json.dumps(
            {
                "trace_schema": "nemotron_voicechat.playout.v1",
                "type": "response.done",
                "response_id": "response-real",
                "status": "completed",
                "client_received_monotonic_s": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    soak.backfill_terminal_identity_from_trace(records, trace)

    assert records[0]["response_id"] == "response-real"
    assert records[0]["terminal_status"] == "completed"
