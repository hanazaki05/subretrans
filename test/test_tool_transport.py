import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from subretrans.providers import (
    ToolDefinition,
    ToolLoopError,
    ToolStepLimitExceeded,
    invoke_with_tools,
)
from subretrans.stats import UsageStats


def tool_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="inspect_context",
            description="Read the bounded subtitle context.",
            input_schema={
                "type": "object",
                "properties": {"cue_id": {"type": "integer"}},
                "required": ["cue_id"],
                "additionalProperties": False,
            },
        ),
    )


def fake_model(*responses: str):
    queue = list(responses)
    calls: list[list[tuple[str, str]]] = []

    def invoke(messages):
        calls.append(list(messages))
        if not queue:
            raise AssertionError("fake model received more requests than expected")
        return AIMessage(
            content=queue.pop(0),
            usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )

    return SimpleNamespace(invoke=invoke), calls


def test_json_action_loop_executes_one_tool_per_step_and_audits() -> None:
    model, calls = fake_model(
        '{"action":"tool_call","name":"inspect_context","arguments":{"cue_id":7}}',
        '{"action":"final","result":{"status":"finish"}}',
    )
    exchanges = []

    result = invoke_with_tools(
        model,
        [("system", "Repair the supplied suggestion."), ("human", "Start.")],
        tool_definitions(),
        lambda name, arguments: {"cue_id": arguments["cue_id"], "text": "context"},
        max_tool_steps=3,
        on_exchange=exchanges.append,
    )

    assert result.final_result == {"status": "finish"}
    assert result.steps == 2
    assert result.usage == UsageStats(4, 6, 10, 0)
    assert tuple(exchanges) == result.exchanges
    assert exchanges[0].action is not None
    assert exchanges[0].action.kind == "tool_call"
    assert exchanges[0].tool_result == {"cue_id": 7, "text": "context"}
    assert exchanges[1].action is not None
    assert exchanges[1].action.kind == "final"
    assert len(calls) == 2
    assert calls[0][0][0] == "system"
    assert "strict-json-actions-v1" in calls[0][0][1]
    assert calls[1][-2][0] == "assistant"
    assert calls[1][-1][0] == "human"
    assert json.loads(calls[1][-1][1]) == {
        "tool_result": {
            "name": "inspect_context",
            "ok": True,
            "result": {"cue_id": 7, "text": "context"},
            "error": None,
        }
    }


def test_invalid_actions_consume_steps_and_return_canonical_protocol_feedback() -> None:
    model, calls = fake_model(
        "not-json",
        '{"action":"tool_call","name":"unknown","arguments":{}}',
        '{"action":"final","result":{"status":"escalate"}}',
    )
    exchanges = []
    executed: list[str] = []

    result = invoke_with_tools(
        model,
        [("human", "Start.")],
        tool_definitions(),
        lambda name, arguments: executed.append(name) or {},
        max_tool_steps=3,
        on_exchange=exchanges.append,
    )

    assert result.final_result == {"status": "escalate"}
    assert result.steps == 3
    assert [exchange.step for exchange in exchanges] == [1, 2, 3]
    assert exchanges[0].error == "response must be exactly one JSON object"
    assert exchanges[1].error == "unknown tool: unknown"
    assert executed == []
    assert json.loads(calls[1][-1][1])["tool_protocol_error"]["required"] == (
        "exactly one JSON action object"
    )


def test_tool_failure_is_recorded_and_the_model_can_escalate() -> None:
    model, _ = fake_model(
        '{"action":"tool_call","name":"inspect_context","arguments":{"cue_id":7}}',
        '{"action":"final","result":{"status":"escalate"}}',
    )

    def fail(name, arguments):
        raise ValueError("context is unavailable")

    result = invoke_with_tools(
        model,
        [("human", "Start.")],
        tool_definitions(),
        fail,
        max_tool_steps=2,
    )

    assert result.final_result == {"status": "escalate"}
    assert result.exchanges[0].error == "context is unavailable"
    assert result.exchanges[0].tool_result == {"message": "context is unavailable"}


def test_action_loop_raises_after_budget_without_silent_completion() -> None:
    model, _ = fake_model("not-json", "still-not-json")
    exchanges = []

    with pytest.raises(ToolStepLimitExceeded) as caught:
        invoke_with_tools(
            model,
            [("human", "Start.")],
            tool_definitions(),
            lambda name, arguments: {},
            max_tool_steps=2,
            on_exchange=exchanges.append,
        )

    assert len(caught.value.exchanges) == 2
    assert caught.value.usage == UsageStats(4, 6, 10, 0)
    assert len(exchanges) == 2


def test_model_transport_failure_is_audited_and_not_retried_by_the_loop() -> None:
    exchanges = []

    def fail(messages):
        raise TimeoutError("provider timeout")

    with pytest.raises(ToolLoopError, match="model invocation failed at tool step 1"):
        invoke_with_tools(
            SimpleNamespace(invoke=fail),
            [("human", "Start.")],
            tool_definitions(),
            lambda name, arguments: {},
            max_tool_steps=4,
            on_exchange=exchanges.append,
        )

    assert len(exchanges) == 1
    assert exchanges[0].step == 1
    assert exchanges[0].response == ""
    assert exchanges[0].error == "provider timeout"


@pytest.mark.parametrize(
    "payload",
    [
        '{"action":"final"}',
        '{"action":"final","result":{},"extra":true}',
        '{"action":"tool_call","name":"inspect_context","arguments":{},"extra":true}',
        '{"action":"tool_call","name":"inspect_context","arguments":[]}',
    ],
)
def test_action_shape_is_strict(payload: str) -> None:
    model, _ = fake_model(payload, '{"action":"final","result":{"status":"escalate"}}')
    result = invoke_with_tools(
        model,
        [("human", "Start.")],
        tool_definitions(),
        lambda name, arguments: {},
        max_tool_steps=2,
    )

    assert result.final_result == {"status": "escalate"}
    assert result.exchanges[0].action is None
    assert result.exchanges[0].error
