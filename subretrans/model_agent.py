"""Strict model-backed semantic QA for bilingual subtitle pairs."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import AIMessage

from .config import RoleModelSettings
from .pairs import SubtitlePair
from .providers import build_chat_model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRepair:
    """A targeted replacement for one Chinese subtitle."""

    id: int
    translation: str


@dataclass(frozen=True)
class AgentRepairHistory:
    """One previously applied repair supplied to a later QA pass."""

    attempt: int
    id: int
    before: str
    after: str


@dataclass(frozen=True)
class AgentQATerm:
    """One read-only terminology mapping from refine memory."""

    eng: str
    zh: str


@dataclass(frozen=True)
class AgentQAGlossaryTerm:
    """One learned terminology entry from refine memory."""

    eng: str
    zh: str
    type: str | None
    confidence: float | None
    evidence_ids: tuple[int, ...]


@dataclass(frozen=True)
class AgentQAMemory:
    """Read-only episode context supplied to every semantic-QA window."""

    story_description: str
    user_glossary: tuple[AgentQATerm, ...]
    glossary: tuple[AgentQAGlossaryTerm, ...]


@dataclass(frozen=True)
class AgentQAResult:
    """Strict semantic-QA decision returned by the agent model."""

    passed: bool
    issues: tuple[str, ...]
    repairs: tuple[AgentRepair, ...]


AgentQA = Callable[
    [
        tuple[SubtitlePair, ...] | list[SubtitlePair],
        str,
        tuple[AgentRepairHistory, ...],
        AgentQAMemory,
    ],
    AgentQAResult,
]


def _exact_object(
    value: object, expected_fields: set[str], *, location: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{location} must be a JSON object")
    fields = set(value)
    if fields != expected_fields:
        missing = sorted(expected_fields - fields)
        unknown = sorted(fields - expected_fields)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{location} has invalid fields ({'; '.join(details)})")
    return value


def build_agent_qa(settings: RoleModelSettings, system_prompt: str) -> AgentQA:
    """Build a semantic-QA callable with strict JSON response validation."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("agent QA system prompt must be a non-empty string")
    model = build_chat_model(settings.config)

    def agent_qa(
        pairs: tuple[SubtitlePair, ...] | list[SubtitlePair],
        structural_qa: str,
        repair_history: tuple[AgentRepairHistory, ...],
        episode_memory: AgentQAMemory,
    ) -> AgentQAResult:
        if type(pairs) not in (tuple, list):
            raise TypeError("pairs must be a tuple or list")
        if any(not isinstance(pair, SubtitlePair) for pair in pairs):
            raise TypeError("pairs must contain only SubtitlePair values")
        if type(structural_qa) is not str:
            raise TypeError("structural_qa must be a string")
        if type(repair_history) is not tuple or any(
            not isinstance(entry, AgentRepairHistory) for entry in repair_history
        ):
            raise TypeError("repair_history must contain AgentRepairHistory values")
        if not isinstance(episode_memory, AgentQAMemory):
            raise TypeError("episode_memory must be an AgentQAMemory")

        input_ids = {pair.id for pair in pairs}
        request_payload = {
            "pairs": [
                {"id": pair.id, "english": pair.eng, "chinese": pair.chinese}
                for pair in pairs
            ],
            "structural_qa": structural_qa,
            "repair_history": [
                {
                    "attempt": entry.attempt,
                    "id": entry.id,
                    "before": entry.before,
                    "after": entry.after,
                }
                for entry in repair_history
            ],
            "episode_memory": {
                "story_description": episode_memory.story_description,
                "user_glossary": [
                    {"eng": entry.eng, "zh": entry.zh}
                    for entry in episode_memory.user_glossary
                ],
                "glossary": [
                    {
                        "eng": entry.eng,
                        "zh": entry.zh,
                        **({"type": entry.type} if entry.type is not None else {}),
                        **(
                            {"confidence": entry.confidence}
                            if entry.confidence is not None
                            else {}
                        ),
                        **(
                            {"evidence_ids": list(entry.evidence_ids)}
                            if entry.evidence_ids
                            else {}
                        ),
                    }
                    for entry in episode_memory.glossary
                ],
            },
        }
        response = model.invoke(
            [
                ("system", system_prompt),
                ("human", json.dumps(request_payload, ensure_ascii=False)),
            ]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("model response must be an AIMessage")
        response_text = response.text
        if not isinstance(response_text, str):
            raise TypeError("AIMessage text must be a string")
        logger.debug(
            "Agent QA raw response: content=%r metadata=%r",
            response.content,
            response.response_metadata,
        )
        if not response_text.strip():
            stop_reason = response.response_metadata.get("stop_reason", "unknown")
            raise ValueError(
                f"agent QA returned no text content (stop_reason={stop_reason})"
            )

        payload = _exact_object(
            json.loads(response_text),
            {"passed", "issues", "repairs"},
            location="response",
        )
        passed = payload["passed"]
        raw_issues = payload["issues"]
        raw_repairs = payload["repairs"]
        if type(passed) is not bool:
            raise ValueError("response.passed must be a boolean")
        if type(raw_issues) is not list:
            raise ValueError("response.issues must be a JSON array")
        if type(raw_repairs) is not list:
            raise ValueError("response.repairs must be a JSON array")

        issues: list[str] = []
        for index, issue in enumerate(raw_issues):
            if type(issue) is not str or not issue.strip():
                raise ValueError(
                    f"response.issues[{index}] must be a non-empty string"
                )
            issues.append(issue)

        repairs: list[AgentRepair] = []
        repair_ids: set[int] = set()
        for index, raw_repair in enumerate(raw_repairs):
            repair = _exact_object(
                raw_repair,
                {"id", "translation"},
                location=f"response.repairs[{index}]",
            )
            repair_id = repair["id"]
            translation = repair["translation"]
            if type(repair_id) is not int:
                raise ValueError(f"response.repairs[{index}].id must be an integer")
            if repair_id not in input_ids:
                raise ValueError(
                    f"response.repairs[{index}].id does not belong to the input"
                )
            if repair_id in repair_ids:
                raise ValueError("response repair ids must be unique")
            if type(translation) is not str or not translation.strip():
                raise ValueError(
                    f"response.repairs[{index}].translation must be a "
                    "non-empty string"
                )
            repair_ids.add(repair_id)
            repairs.append(AgentRepair(repair_id, translation))

        if passed and (issues or repairs):
            raise ValueError("a passing response must have no issues or repairs")
        if not passed and not issues:
            raise ValueError("a failing response must have at least one issue")

        return AgentQAResult(passed, tuple(issues), tuple(repairs))

    return agent_qa
