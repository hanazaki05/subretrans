import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from subretrans.model_agent import (
    AgentQADecisionHistory,
    AgentQAEvidence,
    AgentQAGlossaryTerm,
    AgentQAMemory,
    AgentQAResult,
    AgentQASuggestion,
    AgentQATerm,
    AgentQATranslation,
    build_agent_qa,
)
from subretrans.pairs import SubtitlePair


def _settings() -> Mock:
    settings = Mock()
    settings.config = Mock(name="config")
    return settings


def _pairs() -> tuple[SubtitlePair, ...]:
    return (
        SubtitlePair(4, "He did not leave.", "他走了。"),
        SubtitlePair(5, "Stay here.", "留在这里。"),
    )


def _memory() -> AgentQAMemory:
    return AgentQAMemory(
        "Harm briefs Mac.",
        (AgentQATerm("Harm", "哈姆"),),
        (AgentQAGlossaryTerm("SecNav", "海军部长", "title", 0.9, (4,)),),
    )


@patch("subretrans.model_agent.build_chat_model")
def test_qa_returns_structured_read_only_suggestions(build_chat_model) -> None:
    model = build_chat_model.return_value
    model.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "passed": False,
                "suggestions": [
                    {
                        "affected_ids": [4],
                        "kind": "meaning",
                        "diagnosis": "The negation is reversed.",
                        "evidence": [
                            {"affected_ids": [4], "observation": "did not is negative"}
                        ],
                        "suggested_translations": [
                            {"id": 4, "translation": "他没有离开。"}
                        ],
                    }
                ],
            }
        )
    )
    qa = build_agent_qa(_settings(), "QA prompt")
    result = qa(
        _pairs(),
        "structurally valid",
        (AgentQADecisionHistory("old-key", "dismissed", "false positive"),),
        _memory(),
    )

    assert result == AgentQAResult(
        False,
        (
            AgentQASuggestion(
                (4,),
                "meaning",
                "The negation is reversed.",
                (AgentQAEvidence((4,), "did not is negative"),),
                (AgentQATranslation(4, "他没有离开。"),),
            ),
        ),
    )
    request = json.loads(model.invoke.call_args.args[0][1][1])
    assert request["decision_history"] == [
        {"issue_key": "old-key", "status": "dismissed", "reason": "false positive"}
    ]
    assert "repairs" not in model.invoke.return_value.content


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"passed": True, "suggestions": [], "extra": 1}, "unknown fields"),
        ({"passed": False, "suggestions": []}, "at least one suggestion"),
        (
            {
                "passed": True,
                "suggestions": [
                    {
                        "affected_ids": [4],
                        "kind": "meaning",
                        "diagnosis": "wrong",
                        "evidence": [{"affected_ids": [4], "observation": "evidence"}],
                    }
                ],
            },
            "passing response",
        ),
        (
            {
                "passed": False,
                "suggestions": [
                    {
                        "affected_ids": [9],
                        "kind": "meaning",
                        "diagnosis": "wrong",
                        "evidence": [{"affected_ids": [4], "observation": "evidence"}],
                    }
                ],
            },
            "does not belong",
        ),
        (
            {
                "passed": False,
                "suggestions": [
                    {
                        "affected_ids": [4, 5],
                        "kind": "meaning",
                        "diagnosis": "wrong",
                        "evidence": [{"affected_ids": [4], "observation": "evidence"}],
                        "suggested_translations": [{"id": 4, "translation": "正确"}],
                    }
                ],
            },
            "cover every affected id",
        ),
    ],
)
@patch("subretrans.model_agent.build_chat_model")
def test_qa_rejects_invalid_suggestion_schema(build_chat_model, payload, match) -> None:
    build_chat_model.return_value.invoke.return_value = AIMessage(content=json.dumps(payload))
    qa = build_agent_qa(_settings(), "QA prompt")
    with pytest.raises(ValueError, match=match):
        qa(_pairs(), "valid", (), _memory())


@patch("subretrans.model_agent.build_chat_model")
def test_qa_requires_decision_history_type(build_chat_model) -> None:
    qa = build_agent_qa(_settings(), "QA prompt")
    with pytest.raises(TypeError, match="decision_history"):
        qa(_pairs(), "valid", [], _memory())  # type: ignore[arg-type]
