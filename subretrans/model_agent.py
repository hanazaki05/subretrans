"""Strict model-backed semantic QA for bilingual subtitle pairs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import AIMessage

from .config import RoleModelSettings
from .pairs import SubtitlePair
from .providers import build_chat_model


@dataclass(frozen=True)
class AgentRepair:
    """A targeted replacement for one Chinese subtitle."""

    id: int
    translation: str


@dataclass(frozen=True)
class AgentQAResult:
    """Strict semantic-QA decision returned by the agent model."""

    passed: bool
    issues: tuple[str, ...]
    repairs: tuple[AgentRepair, ...]


AgentQA = Callable[[tuple[SubtitlePair, ...] | list[SubtitlePair], str], AgentQAResult]


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


def build_agent_qa(settings: RoleModelSettings) -> AgentQA:
    """Build a semantic-QA callable with strict JSON response validation."""

    model = build_chat_model(settings.config)
    system_prompt = (
        "Audit the supplied English-Chinese subtitle pairs for semantic "
        "completeness, accuracy, and cross-pair consistency. Treat the local "
        "structural QA conclusion as evidence about subtitle structure, not as "
        "a semantic verdict. Report every material semantic problem as a "
        "specific, non-empty issue. Propose a repair only when a targeted "
        "replacement of an input pair's complete Chinese translation is "
        "needed; preserve its input id exactly.\n"
        "The user message is one JSON object with exactly the keys \"pairs\" "
        "and \"structural_qa\". Each pairs element has exactly the keys "
        "\"id\", \"english\", and \"chinese\". Return JSON only: one object "
        "with exactly the keys \"passed\", \"issues\", and \"repairs\". "
        "passed must be a boolean. issues must be an array of non-empty "
        "strings. repairs must be an array whose elements contain exactly "
        "the keys \"id\" and \"translation\", with an integer input id and "
        "a non-empty complete Chinese translation. Repair ids must be unique. "
        "When passed is true, issues and repairs must both be empty. When "
        "passed is false, issues must be non-empty; repairs may be empty."
    )

    def agent_qa(
        pairs: tuple[SubtitlePair, ...] | list[SubtitlePair], structural_qa: str
    ) -> AgentQAResult:
        if type(pairs) not in (tuple, list):
            raise TypeError("pairs must be a tuple or list")
        if any(not isinstance(pair, SubtitlePair) for pair in pairs):
            raise TypeError("pairs must contain only SubtitlePair values")
        if type(structural_qa) is not str:
            raise TypeError("structural_qa must be a string")

        input_ids = {pair.id for pair in pairs}
        request_payload = {
            "pairs": [
                {"id": pair.id, "english": pair.eng, "chinese": pair.chinese}
                for pair in pairs
            ],
            "structural_qa": structural_qa,
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
