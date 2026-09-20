"""Strict, read-only model-backed semantic QA for bilingual subtitle pairs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .config import RoleModelSettings
from .fsutil import require_exact_fields
from .pairs import SubtitlePair
from .providers import build_chat_model, clean_response_text, invoke_text


@dataclass(frozen=True)
class AgentQATranslation:
    """A non-binding complete Chinese translation suggested for one cue."""

    id: int
    translation: str


@dataclass(frozen=True)
class AgentQAEvidence:
    """Structured evidence for one QA suggestion."""

    affected_ids: tuple[int, ...]
    observation: str


@dataclass(frozen=True)
class AgentQASuggestion:
    """One read-only QA finding; it is never an executable repair."""

    affected_ids: tuple[int, ...]
    kind: str
    diagnosis: str
    evidence: tuple[AgentQAEvidence, ...]
    suggested_translations: tuple[AgentQATranslation, ...] = ()


@dataclass(frozen=True)
class AgentQADecisionHistory:
    """A prior host/repair decision supplied to QA to avoid reopening it blindly."""

    issue_key: str
    status: str
    reason: str


@dataclass(frozen=True)
class AgentQATerm:
    eng: str
    zh: str


@dataclass(frozen=True)
class AgentQAGlossaryTerm:
    """One validated entry from the frozen effective glossary."""

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
    """Strict semantic-QA findings returned by the auditor model."""

    passed: bool
    suggestions: tuple[AgentQASuggestion, ...]


AgentQA = Callable[
    [
        tuple[SubtitlePair, ...] | list[SubtitlePair],
        str,
        tuple[AgentQADecisionHistory, ...],
        AgentQAMemory,
    ],
    AgentQAResult,
]


def _glossary_payload(entry: AgentQAGlossaryTerm) -> dict[str, object]:
    payload: dict[str, object] = {"eng": entry.eng, "zh": entry.zh}
    if entry.type is not None:
        payload["type"] = entry.type
    if entry.confidence is not None:
        payload["confidence"] = entry.confidence
    if entry.evidence_ids:
        payload["evidence_ids"] = list(entry.evidence_ids)
    return payload


def _nonempty_string(value: object, location: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _input_ids(value: object, input_ids: set[int], location: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{location} must be a non-empty JSON array")
    ids: list[int] = []
    for index, cue_id in enumerate(value):
        if type(cue_id) is not int:
            raise ValueError(f"{location}[{index}] must be an integer")
        if cue_id not in input_ids:
            raise ValueError(f"{location}[{index}] does not belong to the input")
        if cue_id in ids:
            raise ValueError(f"{location} ids must be unique")
        ids.append(cue_id)
    return tuple(ids)


def build_agent_qa(settings: RoleModelSettings, system_prompt: str) -> AgentQA:
    """Build a read-only semantic-QA callable with strict JSON validation."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("agent QA system prompt must be a non-empty string")
    model = build_chat_model(settings.config)

    def agent_qa(
        pairs: tuple[SubtitlePair, ...] | list[SubtitlePair],
        structural_qa: str,
        decision_history: tuple[AgentQADecisionHistory, ...],
        episode_memory: AgentQAMemory,
    ) -> AgentQAResult:
        if type(pairs) not in (tuple, list):
            raise TypeError("pairs must be a tuple or list")
        if any(not isinstance(pair, SubtitlePair) for pair in pairs):
            raise TypeError("pairs must contain only SubtitlePair values")
        if type(structural_qa) is not str:
            raise TypeError("structural_qa must be a string")
        if type(decision_history) is not tuple or any(
            not isinstance(entry, AgentQADecisionHistory) for entry in decision_history
        ):
            raise TypeError("decision_history must contain AgentQADecisionHistory values")
        if not isinstance(episode_memory, AgentQAMemory):
            raise TypeError("episode_memory must be an AgentQAMemory")

        input_ids = {pair.id for pair in pairs}
        request_payload = {
            "pairs": [
                {"id": pair.id, "english": pair.eng, "chinese": pair.chinese}
                for pair in pairs
            ],
            "structural_qa": structural_qa,
            "decision_history": [
                {
                    "issue_key": entry.issue_key,
                    "status": entry.status,
                    "reason": entry.reason,
                }
                for entry in decision_history
            ],
            "episode_memory": {
                "story_description": episode_memory.story_description,
                "user_glossary": [
                    {"eng": entry.eng, "zh": entry.zh}
                    for entry in episode_memory.user_glossary
                ],
                "glossary": [
                    _glossary_payload(entry) for entry in episode_memory.glossary
                ],
            },
        }
        response_text, _ = invoke_text(
            model,
            [
                ("system", system_prompt),
                ("human", json.dumps(request_payload, ensure_ascii=False)),
            ],
        )
        payload = require_exact_fields(
            json.loads(clean_response_text(response_text)),
            {"passed", "suggestions"},
            location="response",
        )
        passed = payload["passed"]
        raw_suggestions = payload["suggestions"]
        if type(passed) is not bool:
            raise ValueError("response.passed must be a boolean")
        if type(raw_suggestions) is not list:
            raise ValueError("response.suggestions must be a JSON array")

        suggestions: list[AgentQASuggestion] = []
        for suggestion_index, raw_suggestion in enumerate(raw_suggestions):
            location = f"response.suggestions[{suggestion_index}]"
            if type(raw_suggestion) is not dict:
                raise ValueError(f"{location} must be a JSON object")
            fields = set(raw_suggestion)
            required = {"affected_ids", "kind", "diagnosis", "evidence"}
            allowed = required | {"suggested_translations"}
            if not required.issubset(fields) or not fields.issubset(allowed):
                missing = sorted(required - fields)
                unknown = sorted(fields - allowed)
                details: list[str] = []
                if missing:
                    details.append(f"missing fields: {', '.join(missing)}")
                if unknown:
                    details.append(f"unknown fields: {', '.join(unknown)}")
                raise ValueError(f"{location} has invalid fields ({'; '.join(details)})")
            affected_ids = _input_ids(
                raw_suggestion["affected_ids"], input_ids, f"{location}.affected_ids"
            )
            kind = _nonempty_string(raw_suggestion["kind"], f"{location}.kind")
            diagnosis = _nonempty_string(
                raw_suggestion["diagnosis"], f"{location}.diagnosis"
            )

            raw_evidence = raw_suggestion["evidence"]
            if type(raw_evidence) is not list or not raw_evidence:
                raise ValueError(f"{location}.evidence must be a non-empty JSON array")
            evidence: list[AgentQAEvidence] = []
            for evidence_index, raw_entry in enumerate(raw_evidence):
                evidence_location = f"{location}.evidence[{evidence_index}]"
                entry = require_exact_fields(
                    raw_entry,
                    {"affected_ids", "observation"},
                    location=evidence_location,
                )
                evidence.append(
                    AgentQAEvidence(
                        _input_ids(
                            entry["affected_ids"],
                            input_ids,
                            f"{evidence_location}.affected_ids",
                        ),
                        _nonempty_string(
                            entry["observation"], f"{evidence_location}.observation"
                        ),
                    )
                )

            raw_translations = raw_suggestion.get("suggested_translations", [])
            if type(raw_translations) is not list:
                raise ValueError(
                    f"{location}.suggested_translations must be a JSON array"
                )
            translations: list[AgentQATranslation] = []
            translation_ids: set[int] = set()
            for translation_index, raw_translation in enumerate(raw_translations):
                translation_location = (
                    f"{location}.suggested_translations[{translation_index}]"
                )
                entry = require_exact_fields(
                    raw_translation,
                    {"id", "translation"},
                    location=translation_location,
                )
                cue_id = entry["id"]
                if type(cue_id) is not int:
                    raise ValueError(f"{translation_location}.id must be an integer")
                if cue_id not in affected_ids:
                    raise ValueError(
                        f"{translation_location}.id must belong to affected_ids"
                    )
                if cue_id in translation_ids:
                    raise ValueError("suggested translation ids must be unique")
                translation_ids.add(cue_id)
                translations.append(
                    AgentQATranslation(
                        cue_id,
                        _nonempty_string(
                            entry["translation"], f"{translation_location}.translation"
                        ),
                    )
                )
            if translations and translation_ids != set(affected_ids):
                raise ValueError(
                    f"{location}.suggested_translations must cover every affected id"
                )
            suggestions.append(
                AgentQASuggestion(
                    affected_ids,
                    kind,
                    diagnosis,
                    tuple(evidence),
                    tuple(translations),
                )
            )

        if passed and suggestions:
            raise ValueError("a passing response must have no suggestions")
        if not passed and not suggestions:
            raise ValueError("a failing response must have at least one suggestion")
        return AgentQAResult(passed, tuple(suggestions))

    return agent_qa
