import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from subretrans.model_agent import (
    AgentQAMemory,
    AgentQAGlossaryTerm,
    AgentQAResult,
    AgentQATerm,
    AgentRepair,
    AgentRepairHistory,
    build_agent_qa,
)
from subretrans.pairs import SubtitlePair


QA_PROMPT = (
    "SHARED RULE SENTINEL\n\nQA TASK SENTINEL\n"
    "For adjacent cross-line sentences, return coordinated complete replacements."
)


def role_settings() -> Mock:
    settings = Mock()
    settings.config = Mock(name="config")
    settings.max_output_tokens = 1400
    settings.reasoning_effort = "high"
    settings.temperature = 0.2
    return settings


def pairs() -> tuple[SubtitlePair, ...]:
    return (
        SubtitlePair(id=4, eng="He did not leave.", chinese="他走了。"),
        SubtitlePair(id=9, eng="Stay here.", chinese="留在这里。"),
    )


def episode_memory() -> AgentQAMemory:
    return AgentQAMemory(
        story_description="Harm briefs Mac about the case.",
        user_glossary=(AgentQATerm("Harm", "哈姆"),),
        glossary=(
            AgentQAGlossaryTerm(
                "SecNav", "海军部长", "title", 0.9, (12, 18)
            ),
        ),
    )


@patch("subretrans.model_agent.build_chat_model")
def test_builds_model_and_parses_strict_semantic_qa(build_chat_model) -> None:
    model = build_chat_model.return_value
    model.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "passed": False,
                "issues": ["ID 4 reverses the negation."],
                "repairs": [{"id": 4, "translation": "他没有离开。"}],
            }
        )
    )
    settings = role_settings()

    qa = build_agent_qa(settings, QA_PROMPT)
    result = qa(
        pairs(),
        "Structural QA passed: all events are paired.",
        (),
        episode_memory(),
    )

    assert result == AgentQAResult(
        passed=False,
        issues=("ID 4 reverses the negation.",),
        repairs=(AgentRepair(id=4, translation="他没有离开。"),),
    )
    build_chat_model.assert_called_once_with(settings.config)
    messages = model.invoke.call_args.args[0]
    assert messages[0][1] == QA_PROMPT
    assert messages[0][1].count("SHARED RULE SENTINEL") == 1
    assert messages[0][1].count("QA TASK SENTINEL") == 1
    assert "coordinated complete replacements" in messages[0][1]
    assert json.loads(messages[1][1]) == {
        "pairs": [
            {"id": 4, "english": "He did not leave.", "chinese": "他走了。"},
            {"id": 9, "english": "Stay here.", "chinese": "留在这里。"},
        ],
        "structural_qa": "Structural QA passed: all events are paired.",
        "repair_history": [],
        "episode_memory": {
            "story_description": "Harm briefs Mac about the case.",
            "user_glossary": [{"eng": "Harm", "zh": "哈姆"}],
            "glossary": [
                {
                    "eng": "SecNav",
                    "zh": "海军部长",
                    "type": "title",
                    "confidence": 0.9,
                    "evidence_ids": [12, 18],
                }
            ],
        },
    }
    assert model.invoke.call_args.kwargs == {}


@pytest.mark.parametrize(
    "payload, match",
    [
        (
            {"passed": True, "issues": [], "repairs": [], "extra": True},
            "unknown fields",
        ),
        (
            {
                "passed": False,
                "issues": ["Wrong meaning"],
                "repairs": [
                    {"id": 4, "translation": "正确"},
                    {"id": 4, "translation": "也正确"},
                ],
            },
            "must be unique",
        ),
        (
            {
                "passed": False,
                "issues": ["Wrong meaning"],
                "repairs": [{"id": 7, "translation": "正确"}],
            },
            "does not belong",
        ),
        (
            {"passed": True, "issues": ["Wrong meaning"], "repairs": []},
            "passing response",
        ),
        (
            {
                "passed": True,
                "issues": [],
                "repairs": [{"id": 4, "translation": "正确"}],
            },
            "passing response",
        ),
        (
            {"passed": False, "issues": [], "repairs": []},
            "failing response",
        ),
        (
            {"passed": False, "issues": ["   "], "repairs": []},
            "non-empty string",
        ),
        (
            {
                "passed": False,
                "issues": ["Wrong meaning"],
                "repairs": [{"id": 4, "translation": "  "}],
            },
            "non-empty string",
        ),
    ],
)
@patch("subretrans.model_agent.build_chat_model")
def test_rejects_invalid_semantic_qa_output(
    build_chat_model, payload, match
) -> None:
    build_chat_model.return_value.invoke.return_value = AIMessage(
        content=json.dumps(payload)
    )
    qa = build_agent_qa(role_settings(), QA_PROMPT)

    with pytest.raises(ValueError, match=match):
        qa(pairs(), "Structural QA passed.", (), episode_memory())


@patch("subretrans.model_agent.build_chat_model")
def test_rejects_empty_agent_text_with_stop_reason(build_chat_model) -> None:
    build_chat_model.return_value.invoke.return_value = AIMessage(
        content=[{"type": "thinking", "thinking": "still auditing"}],
        response_metadata={"stop_reason": "max_tokens"},
    )
    qa = build_agent_qa(role_settings(), QA_PROMPT)

    with pytest.raises(ValueError, match="no text content.*max_tokens"):
        qa(pairs(), "Structural QA passed.", (), episode_memory())


@patch("subretrans.model_agent.build_chat_model")
def test_supplies_window_repair_history(build_chat_model) -> None:
    model = build_chat_model.return_value
    model.invoke.return_value = AIMessage(
        content='{"passed":true,"issues":[],"repairs":[]}'
    )
    qa = build_agent_qa(role_settings(), QA_PROMPT)

    qa(
        pairs(),
        "passed",
        (AgentRepairHistory(1, 4, "他走了。", "他没有离开。"),),
        episode_memory(),
    )

    assert json.loads(model.invoke.call_args.args[0][1][1])["repair_history"] == [
        {
            "attempt": 1,
            "id": 4,
            "before": "他走了。",
            "after": "他没有离开。",
        }
    ]


@patch("subretrans.model_agent.build_chat_model")
def test_requires_episode_memory(build_chat_model) -> None:
    qa = build_agent_qa(role_settings(), QA_PROMPT)

    with pytest.raises(TypeError, match="episode_memory"):
        qa(pairs(), "passed")
