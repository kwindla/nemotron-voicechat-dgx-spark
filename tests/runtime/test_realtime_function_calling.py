import asyncio
import json

import pytest

from nemotron_voicechat_runtime.server import (
    ExternalToolBridge,
    coerce_tool_arguments,
    normalize_tool_definitions,
)


def test_normalizes_openai_and_nvidia_tool_schemas() -> None:
    tools = normalize_tool_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup-weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            },
            {"name": "calculate", "parameters": {"type": "object"}},
        ]
    )
    assert [tool["name"] for tool in tools] == ["lookup-weather", "calculate"]
    assert tools[0]["parameters"]["properties"]["city"]["type"] == "string"


def test_rejects_registry_name_collisions() -> None:
    with pytest.raises(ValueError, match="duplicate function name"):
        normalize_tool_definitions(
            [
                {"name": "lookup-weather"},
                {"name": "lookup_weather"},
            ]
        )


def test_external_tool_bridge_round_trip() -> None:
    async def scenario() -> None:
        calls = []
        call_ready = asyncio.Event()

        async def emit(call) -> None:
            calls.append(call)
            call_ready.set()

        bridge = ExternalToolBridge(asyncio.get_running_loop(), emit, timeout_seconds=2.0)
        handler = bridge.handlers([{"name": "lookup-weather"}])["lookup_weather"]
        result_task = asyncio.create_task(
            asyncio.to_thread(handler, {"city": "Paris", "units": "metric"})
        )
        await asyncio.wait_for(call_ready.wait(), timeout=1.0)
        call = calls[0]
        assert call.name == "lookup-weather"
        assert json.loads(call.arguments) == {"city": "Paris", "units": "metric"}
        bridge.submit(call.call_id, '{"temperature":21}')
        assert await result_task == '{"temperature":21}'
        bridge.close()

    asyncio.run(scenario())


def test_external_tool_bridge_restores_declared_numeric_argument_types() -> None:
    async def scenario() -> None:
        calls = []
        call_ready = asyncio.Event()

        async def emit(call) -> None:
            calls.append(call)
            call_ready.set()

        bridge = ExternalToolBridge(asyncio.get_running_loop(), emit, timeout_seconds=2.0)
        handler = bridge.handlers(
            [
                {
                    "name": "random_number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "integer"},
                            "max": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                }
            ]
        )["random_number"]
        result_task = asyncio.create_task(
            asyncio.to_thread(
                handler,
                {"min": "10", "max": "20", "ratio": "0.5", "enabled": "true"},
            )
        )
        await asyncio.wait_for(call_ready.wait(), timeout=1.0)
        assert json.loads(calls[0].arguments) == {
            "enabled": True,
            "max": 20,
            "min": 10,
            "ratio": 0.5,
        }
        bridge.submit(calls[0].call_id, '{"result":17}')
        assert await result_task == '{"result":17}'
        bridge.close()

    asyncio.run(scenario())


def test_tool_argument_coercion_is_lossless_and_schema_scoped() -> None:
    schema = {
        "type": "object",
        "properties": {
            "valid": {"type": "integer"},
            "leading_zero": {"type": "integer"},
            "label": {"type": "string"},
        },
    }
    assert coerce_tool_arguments(
        {"valid": "-12", "leading_zero": "012", "label": "12", "extra": "34"},
        schema,
    ) == {
        "valid": -12,
        "leading_zero": "012",
        "label": "12",
        "extra": "34",
    }


def test_function_output_requires_live_call_id() -> None:
    async def emit(_call) -> None:
        return None

    async def scenario() -> None:
        bridge = ExternalToolBridge(asyncio.get_running_loop(), emit)
        with pytest.raises(ValueError, match="unknown or completed call_id"):
            bridge.submit("call_missing", "{}")
        bridge.close()

    asyncio.run(scenario())
