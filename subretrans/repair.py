"""Bounded autonomous subtitle repair with host-owned validation and checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RepairSettings, RoleModelSettings
from .fsutil import atomic_copy, atomic_write_json, require_exact_fields, sha256_file
from .glossary_validation import (
    GlossaryTerm,
    matched_authoritative_terms,
    normalize_term_key,
    term_occurs,
)
from .model_agent import AgentQASuggestion
from .pairs import SubtitlePair
from .providers import (
    JSONValue,
    ToolDefinition,
    ToolExchange,
    ToolLoopError,
    ToolLoopResult,
    build_chat_model,
    invoke_with_tools,
)
from .reference_reader import ReferenceReader
from .subtitle_processing import postprocess_chinese_cue


REPAIR_STATE_VERSION = 1
_ASS_TAG_RE = re.compile(r"\{[^{}]*\}|</?[^<>]+>")
_ISSUE_STATES = {"open", "merged", "dismissed", "kept", "resolved", "escalated"}
WebFetchExecutor = Callable[[str], JSONValue]


@dataclass(frozen=True)
class RepairSuggestion:
    """A host-identified, provenance-bound read-only QA suggestion."""

    issue_id: str
    key: str
    affected_ids: tuple[int, ...]
    kind: str
    diagnosis: str
    evidence: tuple[dict[str, JSONValue], ...]
    suggested_translations: tuple[dict[str, JSONValue], ...]
    source_window: tuple[int, int]
    source_pass: int
    artifact_hash: str
    effective_glossary_hash: str
    manifest_hash: str


@dataclass(frozen=True)
class RepairOutcome:
    """Terminal state returned by one bounded repair invocation."""

    status: str
    current_pairs: tuple[SubtitlePair, ...]
    current_artifact_hash: str
    state_dir: Path
    tool_steps_used: int
    repair_attempts_used: int
    model_result: ToolLoopResult | None
    reason: str | None = None


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pairs_artifact_hash(pairs: Sequence[SubtitlePair]) -> str:
    """Hash all cue content and structural metadata deterministically."""

    return _canonical_hash(
        [
            {
                "id": pair.id,
                "english": pair.eng,
                "chinese": pair.chinese,
                "meta": pair.meta,
            }
            for pair in pairs
        ]
    )


def host_qa_suggestions(
    suggestions: Sequence[AgentQASuggestion],
    *,
    source_window: tuple[int, int],
    source_pass: int,
    artifact_hash: str,
    effective_glossary_hash: str,
    manifest_hash: str,
) -> tuple[RepairSuggestion, ...]:
    """Attach stable host ids and immutable provenance to model suggestions."""

    hosted: list[RepairSuggestion] = []
    for suggestion in suggestions:
        evidence = tuple(
            {
                "affected_ids": list(entry.affected_ids),
                "observation": entry.observation,
            }
            for entry in suggestion.evidence
        )
        translations = tuple(
            {"id": entry.id, "translation": entry.translation}
            for entry in suggestion.suggested_translations
        )
        key_payload = {
            "manifest_hash": manifest_hash,
            "affected_ids": sorted(suggestion.affected_ids),
            "kind": suggestion.kind.strip().casefold(),
            "diagnosis": " ".join(suggestion.diagnosis.split()).casefold(),
            "evidence": evidence,
        }
        key = _canonical_hash(key_payload)
        issue_id_hash = _canonical_hash(
            {
                "key": key,
                "source_window": source_window,
                "source_pass": source_pass,
                "artifact_hash": artifact_hash,
            }
        )
        hosted.append(
            RepairSuggestion(
                issue_id=f"issue-{issue_id_hash[:16]}",
                key=key,
                affected_ids=suggestion.affected_ids,
                kind=suggestion.kind,
                diagnosis=suggestion.diagnosis,
                evidence=evidence,
                suggested_translations=translations,
                source_window=source_window,
                source_pass=source_pass,
                artifact_hash=artifact_hash,
                effective_glossary_hash=effective_glossary_hash,
                manifest_hash=manifest_hash,
            )
        )
    return tuple(hosted)


def _pair_payload(pair: SubtitlePair) -> dict[str, JSONValue]:
    return {
        "id": pair.id,
        "english": pair.eng,
        "chinese": pair.chinese,
        "meta": pair.meta,  # type: ignore[dict-item]
    }


def _pair_from_payload(value: object, location: str) -> SubtitlePair:
    payload = require_exact_fields(
        value, {"id", "english", "chinese", "meta"}, location=location
    )
    if type(payload["id"]) is not int:
        raise ValueError(f"{location}.id must be an integer")
    if type(payload["english"]) is not str or type(payload["chinese"]) is not str:
        raise ValueError(f"{location} text fields must be strings")
    if payload["meta"] is not None and type(payload["meta"]) is not dict:
        raise ValueError(f"{location}.meta must be an object or null")
    return SubtitlePair(
        payload["id"], payload["english"], payload["chinese"], payload["meta"]
    )


def _nonempty_string(value: object, location: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _integer_list(value: object, location: str, *, nonempty: bool = True) -> list[int]:
    if type(value) is not list or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{location} must be a {qualifier}JSON array")
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int:
            raise ValueError(f"{location}[{index}] must be an integer")
        if item in result:
            raise ValueError(f"{location} values must be unique")
        result.append(item)
    return result


def _string_list(value: object, location: str, *, nonempty: bool = True) -> list[str]:
    if type(value) is not list or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{location} must be a {qualifier}JSON array")
    result = [_nonempty_string(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ValueError(f"{location} values must be unique")
    return result


def _exchange_payload(exchange: ToolExchange) -> dict[str, JSONValue]:
    return {
        "step": exchange.step,
        "request": [list(message) for message in exchange.request],
        "response": exchange.response,
        "action": asdict(exchange.action) if exchange.action is not None else None,
        "tool_result": exchange.tool_result,
        "error": exchange.error,
    }


class RepairSession:
    """Mutable host boundary for one resumable repair run."""

    def __init__(
        self,
        *,
        state_dir: Path,
        settings: RepairSettings,
        refined_pairs: Sequence[SubtitlePair],
        current_pairs: Sequence[SubtitlePair],
        suggestions: Sequence[RepairSuggestion],
        operations: Sequence[str],
        episode_replacements: Sequence[tuple[str, str]],
        manifest_hash: str,
        effective_glossary: Mapping[str, Any] | None = None,
        reference_reader: ReferenceReader | None = None,
        webfetch_executor: WebFetchExecutor | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.settings = settings
        self.refined = {pair.id: SubtitlePair(pair.id, pair.eng, pair.chinese, pair.meta) for pair in refined_pairs}
        self.current = {pair.id: SubtitlePair(pair.id, pair.eng, pair.chinese, pair.meta) for pair in current_pairs}
        if set(self.refined) != set(self.current):
            raise ValueError("refined and current artifacts must contain the same cue ids")
        if any(self.refined[cue_id].eng != self.current[cue_id].eng for cue_id in self.current):
            raise ValueError("current artifact must preserve every refined English cue")
        self.suggestions = {entry.issue_id: entry for entry in suggestions}
        if len(self.suggestions) != len(suggestions):
            raise ValueError("suggestion issue ids must be unique")
        if any(entry.manifest_hash != manifest_hash for entry in suggestions):
            raise ValueError("suggestion manifest hash does not match this repair run")
        if any(
            cue_id not in self.current
            for entry in suggestions
            for cue_id in entry.affected_ids
        ):
            raise ValueError("suggestion contains an unknown cue id")
        self.decisions: list[dict[str, JSONValue]] = []
        self.history: list[dict[str, JSONValue]] = []
        self.coverage: list[dict[str, JSONValue]] = []
        self.staged: list[dict[str, JSONValue]] = []
        self.exchanges: list[dict[str, JSONValue]] = []
        self.issue_states = {entry.issue_id: "open" for entry in suggestions}
        self.operations = tuple(operations)
        self.episode_replacements = tuple(episode_replacements)
        self.manifest_hash = _nonempty_string(manifest_hash, "manifest_hash")
        self.effective_glossary: dict[str, Any] | None = None
        self.reference_reader = reference_reader
        self.webfetch_executor = webfetch_executor
        self.tool_steps_used = 0
        self.repair_attempts_used = 0
        self.full_sweeps_completed = 0
        self.terminal_status: str | None = None
        self.terminal_reason: str | None = None
        self._generation = 0
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if effective_glossary is not None:
            self._set_effective_glossary(effective_glossary)
        self._commit()

    @property
    def ordered_current(self) -> tuple[SubtitlePair, ...]:
        return tuple(self.current[cue_id] for cue_id in sorted(self.current))

    @property
    def current_hash(self) -> str:
        return pairs_artifact_hash(self.ordered_current)

    @property
    def covered_ids(self) -> set[int]:
        covered: set[int] = set()
        for entry in self.coverage:
            if entry["completed"] is True:
                covered.update(int(cue_id) for cue_id in entry["covered_ids"])
        return covered & set(self.current)

    def _set_effective_glossary(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("effective_glossary must be a mapping")
        authoritative = value.get("authoritative")
        learned = value.get("learned")
        if type(authoritative) is not list or type(learned) is not list:
            raise ValueError(
                "effective_glossary must contain authoritative and learned arrays"
            )
        for section_name, entries in (
            ("authoritative", authoritative),
            ("learned", learned),
        ):
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"effective_glossary.{section_name}[{index}] must be an object"
                    )
                _nonempty_string(
                    entry.get("eng"),
                    f"effective_glossary.{section_name}[{index}].eng",
                )
                _nonempty_string(
                    entry.get("zh"),
                    f"effective_glossary.{section_name}[{index}].zh",
                )
        self.effective_glossary = dict(value)

    def set_effective_glossary(self, value: Mapping[str, Any]) -> None:
        """Bind the frozen glossary used by host-side group validation."""

        self._set_effective_glossary(value)
        self._commit()

    def _artifact_payloads(self) -> dict[str, dict[str, Any]]:
        payloads: dict[str, dict[str, Any]] = {
            "suggestions.json": {
                "version": REPAIR_STATE_VERSION,
                "suggestions": [asdict(entry) for entry in self.suggestions.values()],
                "issue_states": self.issue_states,
            },
            "decisions.json": {"version": REPAIR_STATE_VERSION, "decisions": self.decisions},
            "history.json": {"version": REPAIR_STATE_VERSION, "history": self.history},
            "coverage.json": {
                "version": REPAIR_STATE_VERSION,
                "full_sweeps_completed": self.full_sweeps_completed,
                "coverage": self.coverage,
            },
            "staged.json": {"version": REPAIR_STATE_VERSION, "groups": self.staged},
            "exchanges.json": {
                "version": REPAIR_STATE_VERSION,
                "exchanges": self.exchanges,
            },
            "refined.json": {
                "version": REPAIR_STATE_VERSION,
                "artifact_hash": pairs_artifact_hash(
                    tuple(self.refined[cue_id] for cue_id in sorted(self.refined))
                ),
                "pairs": [
                    _pair_payload(self.refined[cue_id]) for cue_id in sorted(self.refined)
                ],
            },
            "current.json": {
                "version": REPAIR_STATE_VERSION,
                "artifact_hash": self.current_hash,
                "pairs": [_pair_payload(pair) for pair in self.ordered_current],
            },
        }
        if self.effective_glossary is not None:
            payloads["effective-glossary.json"] = {
                "version": REPAIR_STATE_VERSION,
                "effective_glossary": self.effective_glossary,
            }
        return payloads

    def _commit(self) -> None:
        payloads = self._artifact_payloads()
        generation = self._generation + 1
        generations = self.state_dir / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        generation_dir = generations / f"{generation:08d}"
        while generation_dir.exists():
            generation += 1
            generation_dir = generations / f"{generation:08d}"
        generation_dir.mkdir()
        for name, payload in payloads.items():
            atomic_write_json(generation_dir / name, payload)
        manifest = {
            "version": REPAIR_STATE_VERSION,
            "generation": generation,
            "generation_dir": generation_dir.relative_to(self.state_dir).as_posix(),
            "source_manifest_hash": self.manifest_hash,
            "artifacts": {
                name: sha256_file(generation_dir / name) for name in payloads
            },
            "budgets": {
                "tool_steps_used": self.tool_steps_used,
                "repair_attempts_used": self.repair_attempts_used,
                "full_sweeps_completed": self.full_sweeps_completed,
            },
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
        }
        atomic_write_json(self.state_dir / "repair-state.json", manifest)
        self._generation = generation
        # Root-level files are compatibility views for pipeline export only.
        # Resume always follows the atomic pointer above and never trusts them.
        for name in payloads:
            atomic_copy(generation_dir / name, self.state_dir / name)

    def latest_artifact_path(self, name: str) -> Path:
        """Return one artifact from the last atomically committed generation."""

        pointer = json.loads(
            (self.state_dir / "repair-state.json").read_text(encoding="utf-8")
        )
        artifacts = pointer.get("artifacts")
        if type(artifacts) is not dict or name not in artifacts:
            raise KeyError(f"unknown repair artifact: {name}")
        return self.state_dir / pointer["generation_dir"] / name

    def begin_round(
        self,
        *,
        current_pairs: Sequence[SubtitlePair],
        suggestions: Sequence[RepairSuggestion],
    ) -> dict[str, JSONValue]:
        """Synchronize a new QA round without resetting history or budgets."""

        incoming = {pair.id: pair for pair in current_pairs}
        if set(incoming) != set(self.current):
            raise ValueError("new repair round must preserve the cue id set")
        if any(
            incoming[cue_id].eng != self.current[cue_id].eng
            or incoming[cue_id].meta != self.current[cue_id].meta
            for cue_id in self.current
        ):
            raise ValueError("new repair round must preserve English, timing, tags, and structure")
        changed_ids = {
            cue_id
            for cue_id in self.current
            if incoming[cue_id].chinese != self.current[cue_id].chinese
        }
        self.current = {
            cue_id: SubtitlePair(
                pair.id,
                pair.eng,
                pair.chinese,
                dict(pair.meta) if pair.meta else pair.meta,
            )
            for cue_id, pair in incoming.items()
        }
        invalidated = {
            neighbor
            for cue_id in changed_ids
            for neighbor in range(
                cue_id - self.settings.context_radius,
                cue_id + self.settings.context_radius + 1,
            )
            if neighbor in self.current
        }
        if invalidated:
            for entry in self.coverage:
                if entry["completed"] is True:
                    entry["covered_ids"] = [
                        cue_id
                        for cue_id in entry["covered_ids"]
                        if int(cue_id) not in invalidated
                    ]

        prior_by_key: dict[str, str] = {}
        for issue_id, existing in self.suggestions.items():
            status = self.issue_states[issue_id]
            if status != "open":
                prior_by_key.setdefault(existing.key, status)
        added = 0
        linked = 0
        for suggestion in suggestions:
            if suggestion.manifest_hash != self.manifest_hash:
                raise ValueError("new suggestion manifest hash does not match this run")
            if any(cue_id not in self.current for cue_id in suggestion.affected_ids):
                raise ValueError("new suggestion contains an unknown cue id")
            if suggestion.issue_id in self.suggestions:
                continue
            self.suggestions[suggestion.issue_id] = suggestion
            prior_status = prior_by_key.get(suggestion.key)
            if prior_status in {"dismissed", "kept"}:
                self.issue_states[suggestion.issue_id] = prior_status
                self.decisions.append(
                    {
                        "issue_id": suggestion.issue_id,
                        "status": prior_status,
                        "reason": "linked to unchanged prior decision by stable issue key",
                        "artifact_hash": self.current_hash,
                    }
                )
                linked += 1
            else:
                self.issue_states[suggestion.issue_id] = "open"
            added += 1
        self.terminal_status = None
        self.terminal_reason = None
        self._commit()
        return {
            "added_suggestions": added,
            "linked_prior_decisions": linked,
            "changed_ids": sorted(changed_ids),
            "artifact_hash": self.current_hash,
            "generation": self._generation,
        }

    @classmethod
    def resume(
        cls,
        *,
        state_dir: Path,
        settings: RepairSettings,
        operations: Sequence[str],
        episode_replacements: Sequence[tuple[str, str]],
        manifest_hash: str,
        reference_reader: ReferenceReader | None = None,
        webfetch_executor: WebFetchExecutor | None = None,
    ) -> "RepairSession":
        """Resume a checkpoint without resetting any cumulative budget."""

        root = Path(state_dir)
        manifest = json.loads((root / "repair-state.json").read_text(encoding="utf-8"))
        if manifest.get("version") != REPAIR_STATE_VERSION:
            raise ValueError("unsupported repair state version")
        if manifest.get("source_manifest_hash") != manifest_hash:
            raise ValueError("repair state manifest hash does not match this run")
        raw_artifacts = manifest.get("artifacts")
        if type(raw_artifacts) is not dict:
            raise ValueError("repair state artifacts must be an object")
        generation_dir_value = manifest.get("generation_dir")
        if type(generation_dir_value) is not str or not generation_dir_value:
            raise ValueError("repair state generation_dir must be a non-empty string")
        generation_dir = root / generation_dir_value
        for name, expected_hash in raw_artifacts.items():
            if sha256_file(generation_dir / name) != expected_hash:
                raise ValueError(f"repair checkpoint artifact changed: {name}")

        current_payload = json.loads(
            (generation_dir / "current.json").read_text(encoding="utf-8")
        )
        current_pairs = tuple(
            _pair_from_payload(value, f"current.pairs[{index}]")
            for index, value in enumerate(current_payload["pairs"])
        )
        refined_payload = json.loads(
            (generation_dir / "refined.json").read_text(encoding="utf-8")
        )
        refined_pairs = tuple(
            _pair_from_payload(value, f"refined.pairs[{index}]")
            for index, value in enumerate(refined_payload["pairs"])
        )
        suggestions_payload = json.loads(
            (generation_dir / "suggestions.json").read_text(encoding="utf-8")
        )
        suggestions = tuple(
            RepairSuggestion(
                issue_id=value["issue_id"],
                key=value["key"],
                affected_ids=tuple(value["affected_ids"]),
                kind=value["kind"],
                diagnosis=value["diagnosis"],
                evidence=tuple(value["evidence"]),
                suggested_translations=tuple(value["suggested_translations"]),
                source_window=tuple(value["source_window"]),
                source_pass=value["source_pass"],
                artifact_hash=value["artifact_hash"],
                effective_glossary_hash=value["effective_glossary_hash"],
                manifest_hash=value["manifest_hash"],
            )
            for value in suggestions_payload["suggestions"]
        )
        session = cls.__new__(cls)
        session.state_dir = root
        session.settings = settings
        session.refined = {pair.id: pair for pair in refined_pairs}
        session.current = {pair.id: pair for pair in current_pairs}
        session.suggestions = {entry.issue_id: entry for entry in suggestions}
        session.issue_states = suggestions_payload["issue_states"]
        session.decisions = json.loads(
            (generation_dir / "decisions.json").read_text(encoding="utf-8")
        )["decisions"]
        session.history = json.loads(
            (generation_dir / "history.json").read_text(encoding="utf-8")
        )["history"]
        coverage_payload = json.loads(
            (generation_dir / "coverage.json").read_text(encoding="utf-8")
        )
        session.coverage = coverage_payload["coverage"]
        session.staged = json.loads(
            (generation_dir / "staged.json").read_text(encoding="utf-8")
        )["groups"]
        session.exchanges = json.loads(
            (generation_dir / "exchanges.json").read_text(encoding="utf-8")
        )["exchanges"]
        session.operations = tuple(operations)
        session.episode_replacements = tuple(episode_replacements)
        session.manifest_hash = manifest_hash
        glossary_path = generation_dir / "effective-glossary.json"
        session.effective_glossary = (
            json.loads(glossary_path.read_text(encoding="utf-8"))["effective_glossary"]
            if glossary_path.is_file()
            else None
        )
        session.reference_reader = reference_reader
        session.webfetch_executor = webfetch_executor
        budgets = manifest["budgets"]
        session.tool_steps_used = budgets["tool_steps_used"]
        session.repair_attempts_used = budgets["repair_attempts_used"]
        session.full_sweeps_completed = budgets["full_sweeps_completed"]
        session.terminal_status = manifest["terminal_status"]
        session.terminal_reason = manifest["terminal_reason"]
        session._generation = manifest["generation"]
        return session

    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the complete, finite capability set exposed to the model."""

        obj: dict[str, JSONValue] = {"type": "object"}
        tools = [
            ToolDefinition(
                "inspect_context",
                "Read a window with completed=false, then call the same current window with completed=true after review.",
                {
                    **obj,
                    "properties": {
                        "start_id": {"type": "integer"},
                        "end_id": {"type": "integer"},
                        "completed": {"type": "boolean"},
                    },
                    "required": ["start_id", "end_id", "completed"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "merge_issues",
                "Merge overlapping or duplicate open suggestions while preserving them.",
                {
                    **obj,
                    "properties": {
                        "issue_ids": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["issue_ids", "reason"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "dismiss_issue",
                "Dismiss a QA false positive with an auditable reason.",
                {
                    **obj,
                    "properties": {
                        "issue_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["issue_id", "reason"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "keep_original",
                "Keep the current text for an issue with an auditable reason.",
                {
                    **obj,
                    "properties": {
                        "issue_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["issue_id", "reason"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "open_issue",
                "Open a newly discovered issue during the mandatory full sweep.",
                {
                    **obj,
                    "properties": {
                        "affected_ids": {"type": "array", "items": {"type": "integer"}},
                        "kind": {"type": "string"},
                        "diagnosis": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["affected_ids", "kind", "diagnosis", "evidence"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "stage_group_repair",
                "Validate and atomically apply one complete 1-3 cue repair group.",
                {
                    **obj,
                    "properties": {
                        "group_id": {"type": "string"},
                        "base_artifact_hash": {"type": "string"},
                        "affected_ids": {"type": "array", "items": {"type": "integer"}},
                        "translations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "translation": {"type": "string"},
                                },
                                "required": ["id", "translation"],
                                "additionalProperties": False,
                            },
                        },
                        "issue_ids": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "group_id",
                        "base_artifact_hash",
                        "affected_ids",
                        "translations",
                        "issue_ids",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "finish",
                "Finish only after full coverage and all issues are resolved.",
                {**obj, "properties": {}, "required": [], "additionalProperties": False},
            ),
            ToolDefinition(
                "escalate",
                "Stop safely and send unresolved issues to human review.",
                {
                    **obj,
                    "properties": {
                        "reason": {"type": "string"},
                        "issue_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["reason", "issue_ids"],
                    "additionalProperties": False,
                },
            ),
        ]
        if self.reference_reader is not None:
            tools.extend(
                (
                    ToolDefinition(
                        "list_resources",
                        "List allowed subtitle references below the configured root.",
                        {
                            **obj,
                            "properties": {"limit": {"type": "integer"}},
                            "required": ["limit"],
                            "additionalProperties": False,
                        },
                    ),
                    ToolDefinition(
                        "search_subtitles",
                        "Search allowed subtitle references for a text query.",
                        {
                            **obj,
                            "properties": {
                                "query": {"type": "string"},
                                "paths": {"type": "array", "items": {"type": "string"}},
                                "max_results": {"type": "integer"},
                            },
                            "required": ["query", "paths", "max_results"],
                            "additionalProperties": False,
                        },
                    ),
                    ToolDefinition(
                        "read_subtitle_context",
                        "Read a bounded line window from one allowed subtitle reference.",
                        {
                            **obj,
                            "properties": {
                                "path": {"type": "string"},
                                "line": {"type": "integer"},
                                "radius": {"type": "integer"},
                            },
                            "required": ["path", "line", "radius"],
                            "additionalProperties": False,
                        },
                    ),
                    ToolDefinition(
                        "compare",
                        "Compare two allowed subtitle references with a bounded unified diff.",
                        {
                            **obj,
                            "properties": {
                                "left": {"type": "string"},
                                "right": {"type": "string"},
                                "max_diff_lines": {"type": "integer"},
                            },
                            "required": ["left", "right", "max_diff_lines"],
                            "additionalProperties": False,
                        },
                    ),
                )
            )
        if self.webfetch_executor is not None:
            tools.append(
                ToolDefinition(
                    "webfetch",
                    "Fetch one public HTTPS page through the injected bounded executor.",
                    {
                        **obj,
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                )
            )
        return tuple(tools)

    def _require_issue(self, issue_id: str, *, open_only: bool = True) -> None:
        if issue_id not in self.issue_states:
            raise ValueError(f"unknown issue id: {issue_id}")
        if open_only and self.issue_states[issue_id] != "open":
            raise ValueError(f"issue is not open: {issue_id}")

    def _record_decision(self, issue_id: str, status: str, reason: str, **extra: JSONValue) -> None:
        if status not in _ISSUE_STATES:
            raise ValueError(f"unsupported issue state: {status}")
        self.issue_states[issue_id] = status
        self.decisions.append(
            {
                "issue_id": issue_id,
                "status": status,
                "reason": reason,
                "artifact_hash": self.current_hash,
                **extra,
            }
        )

    def _inspect_context(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments), {"start_id", "end_id", "completed"}, location="inspect_context"
        )
        start_id = values["start_id"]
        end_id = values["end_id"]
        completed = values["completed"]
        if type(start_id) is not int or type(end_id) is not int:
            raise ValueError("inspect_context ids must be integers")
        if type(completed) is not bool:
            raise ValueError("inspect_context.completed must be a boolean")
        ordered_ids = sorted(self.current)
        if start_id not in self.current or end_id not in self.current or start_id > end_id:
            raise ValueError("inspect_context range must name existing ordered cues")
        ids = [cue_id for cue_id in ordered_ids if start_id <= cue_id <= end_id]
        if not ids:
            raise ValueError("inspect_context range is empty")
        if completed and not any(
            entry["start_id"] == start_id
            and entry["end_id"] == end_id
            and entry["artifact_hash"] == self.current_hash
            and entry["completed"] is False
            for entry in self.coverage
        ):
            raise ValueError(
                "inspect_context completed=true requires the same window to be provided first"
            )
        entry: dict[str, JSONValue] = {
            "start_id": start_id,
            "end_id": end_id,
            "artifact_hash": self.current_hash,
            "completed": completed,
            "covered_ids": ids if completed else [],
            "opened_issue_ids": [],
            "group_ids": [],
        }
        self.coverage.append(entry)
        if completed and self.covered_ids == set(self.current):
            if self.full_sweeps_completed == 0:
                self.full_sweeps_completed = 1
        self._commit()
        return {
            "refined": [_pair_payload(self.refined[cue_id]) for cue_id in ids],
            "current": [_pair_payload(self.current[cue_id]) for cue_id in ids],
            "artifact_hash": self.current_hash,
            "completed_recorded": completed,
        }

    def _merge_issues(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments), {"issue_ids", "reason"}, location="merge_issues"
        )
        issue_ids = _string_list(values["issue_ids"], "merge_issues.issue_ids")
        if len(issue_ids) < 2:
            raise ValueError("merge_issues requires at least two issues")
        reason = _nonempty_string(values["reason"], "merge_issues.reason")
        for issue_id in issue_ids:
            self._require_issue(issue_id)
        key = _canonical_hash({"merged": sorted(issue_ids), "manifest_hash": self.manifest_hash})
        merged_id = f"issue-{key[:16]}"
        if merged_id in self.issue_states:
            raise ValueError("this issue merge already exists")
        affected = sorted(
            {
                cue_id
                for issue_id in issue_ids
                for cue_id in self.suggestions[issue_id].affected_ids
            }
        )
        first = self.suggestions[issue_ids[0]]
        self.suggestions[merged_id] = RepairSuggestion(
            merged_id,
            key,
            tuple(affected),
            "merged",
            reason,
            tuple({"source_issue_id": issue_id} for issue_id in issue_ids),
            (),
            first.source_window,
            first.source_pass,
            self.current_hash,
            first.effective_glossary_hash,
            self.manifest_hash,
        )
        self.issue_states[merged_id] = "open"
        for issue_id in issue_ids:
            self._record_decision(issue_id, "merged", reason, merged_into=merged_id)
        self._commit()
        return {"merged_issue_id": merged_id, "affected_ids": affected}

    def _dismiss_or_keep(self, name: str, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments), {"issue_id", "reason"}, location=name
        )
        issue_id = _nonempty_string(values["issue_id"], f"{name}.issue_id")
        reason = _nonempty_string(values["reason"], f"{name}.reason")
        self._require_issue(issue_id)
        status = "dismissed" if name == "dismiss_issue" else "kept"
        self._record_decision(issue_id, status, reason)
        self._commit()
        return {"issue_id": issue_id, "status": status}

    def _open_issue(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments),
            {"affected_ids", "kind", "diagnosis", "evidence"},
            location="open_issue",
        )
        affected_ids = _integer_list(values["affected_ids"], "open_issue.affected_ids")
        if any(cue_id not in self.current for cue_id in affected_ids):
            raise ValueError("open_issue affected_ids must exist in the current artifact")
        kind = _nonempty_string(values["kind"], "open_issue.kind")
        diagnosis = _nonempty_string(values["diagnosis"], "open_issue.diagnosis")
        evidence = _nonempty_string(values["evidence"], "open_issue.evidence")
        key = _canonical_hash(
            {
                "manifest_hash": self.manifest_hash,
                "affected_ids": sorted(affected_ids),
                "kind": kind.casefold(),
                "diagnosis": " ".join(diagnosis.split()).casefold(),
                "evidence": evidence,
            }
        )
        issue_id = f"issue-{key[:16]}"
        if issue_id in self.issue_states:
            return {"issue_id": issue_id, "status": self.issue_states[issue_id]}
        all_ids = sorted(self.current)
        self.suggestions[issue_id] = RepairSuggestion(
            issue_id,
            key,
            tuple(affected_ids),
            kind,
            diagnosis,
            ({"affected_ids": affected_ids, "observation": evidence},),
            (),
            (all_ids[0], all_ids[-1]),
            -1,
            self.current_hash,
            "",
            self.manifest_hash,
        )
        self.issue_states[issue_id] = "open"
        for entry in reversed(self.coverage):
            if entry["completed"] is True and any(
                int(entry["start_id"]) <= cue_id <= int(entry["end_id"])
                for cue_id in affected_ids
            ):
                opened = list(entry["opened_issue_ids"])
                opened.append(issue_id)
                entry["opened_issue_ids"] = opened
                break
        self._commit()
        return {"issue_id": issue_id, "status": "open"}

    def _glossary_violations(
        self, affected_ids: Sequence[int], candidate: Mapping[int, SubtitlePair]
    ) -> tuple[str, ...]:
        if self.effective_glossary is None:
            return ("effective glossary is not bound to the repair session",)
        english = " ".join(self.current[cue_id].eng for cue_id in affected_ids)
        chinese = normalize_term_key(
            " ".join(candidate[cue_id].chinese for cue_id in affected_ids)
        )
        violations: list[str] = []
        authority = tuple(
            GlossaryTerm(
                eng=str(entry["eng"]),
                zh=str(entry["zh"]),
                source="user_glossary",
            )
            for entry in self.effective_glossary["authoritative"]
            if term_occurs(str(entry["eng"]), english)
        )
        for term in matched_authoritative_terms(english, authority):
            if normalize_term_key(term.zh) not in chinese:
                violations.append(f"authoritative:{term.eng}->{term.zh}")
        for entry in self.effective_glossary["learned"]:
            eng = str(entry["eng"])
            zh = str(entry["zh"])
            if term_occurs(eng, english) and normalize_term_key(zh) not in chinese:
                violations.append(f"learned:{eng}->{zh}")
        return tuple(violations)

    def _stage_group_repair(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments),
            {
                "group_id",
                "base_artifact_hash",
                "affected_ids",
                "translations",
                "issue_ids",
                "reason",
            },
            location="stage_group_repair",
        )
        group_id = _nonempty_string(values["group_id"], "stage_group_repair.group_id")
        if any(entry["group_id"] == group_id for entry in self.staged):
            raise ValueError(f"group id already exists: {group_id}")
        base_hash = _nonempty_string(
            values["base_artifact_hash"], "stage_group_repair.base_artifact_hash"
        )
        affected_ids = _integer_list(
            values["affected_ids"], "stage_group_repair.affected_ids"
        )
        if len(affected_ids) > self.settings.max_group_span:
            raise ValueError("repair group exceeds max_group_span")
        if affected_ids != sorted(affected_ids) or any(
            right != left + 1 for left, right in zip(affected_ids, affected_ids[1:])
        ):
            raise ValueError("repair group ids must be consecutive and ordered")
        if any(cue_id not in self.current for cue_id in affected_ids):
            raise ValueError("repair group contains an unknown cue id")
        issue_ids = _string_list(values["issue_ids"], "stage_group_repair.issue_ids")
        for issue_id in issue_ids:
            self._require_issue(issue_id)
        reason = _nonempty_string(values["reason"], "stage_group_repair.reason")

        raw_translations = values["translations"]
        if type(raw_translations) is not list:
            raise ValueError("stage_group_repair.translations must be a JSON array")
        translations: dict[int, str] = {}
        for index, raw_translation in enumerate(raw_translations):
            location = f"stage_group_repair.translations[{index}]"
            entry = require_exact_fields(
                raw_translation, {"id", "translation"}, location=location
            )
            cue_id = entry["id"]
            if type(cue_id) is not int:
                raise ValueError(f"{location}.id must be an integer")
            if cue_id in translations:
                raise ValueError("repair group translation ids must be unique")
            translations[cue_id] = _nonempty_string(
                entry["translation"], f"{location}.translation"
            )
        if set(translations) != set(affected_ids):
            raise ValueError("repair group must provide one complete translation per affected id")

        staged_entry: dict[str, JSONValue] = {
            "group_id": group_id,
            "base_artifact_hash": base_hash,
            "affected_ids": affected_ids,
            "translations": [
                {"id": cue_id, "translation": translations[cue_id]}
                for cue_id in affected_ids
            ],
            "issue_ids": issue_ids,
            "reason": reason,
            "status": "staged",
        }
        self.staged.append(staged_entry)
        if base_hash != self.current_hash:
            staged_entry["status"] = "rejected"
            staged_entry["rejection_reason"] = "stale base artifact hash"
            self.history.append(
                {"kind": "group_rejected", "group_id": group_id, "reason": staged_entry["rejection_reason"]}
            )
            self._commit()
            return {"group_id": group_id, "status": "rejected", "reason": staged_entry["rejection_reason"]}

        if self.effective_glossary is None:
            staged_entry["status"] = "rejected"
            staged_entry["rejection_reason"] = "effective glossary is not bound"
            self.history.append(
                {
                    "kind": "group_rejected",
                    "group_id": group_id,
                    "reason": staged_entry["rejection_reason"],
                }
            )
            self._commit()
            return {
                "group_id": group_id,
                "status": "rejected",
                "reason": staged_entry["rejection_reason"],
            }

        candidate = {
            cue_id: SubtitlePair(pair.id, pair.eng, pair.chinese, dict(pair.meta) if pair.meta else pair.meta)
            for cue_id, pair in self.current.items()
        }
        changes: list[dict[str, JSONValue]] = []
        for cue_id in affected_ids:
            before = self.current[cue_id]
            proposed = translations[cue_id]
            if _ASS_TAG_RE.findall(proposed) != _ASS_TAG_RE.findall(before.chinese):
                staged_entry["status"] = "rejected"
                staged_entry["rejection_reason"] = f"cue {cue_id} changes ASS/HTML tags"
                self.history.append(
                    {"kind": "group_rejected", "group_id": group_id, "reason": staged_entry["rejection_reason"]}
                )
                self._commit()
                return {"group_id": group_id, "status": "rejected", "reason": staged_entry["rejection_reason"]}

            without_episode = postprocess_chinese_cue(
                proposed,
                tuple(op for op in self.operations if op != "episode_replacements"),
                (),
            )
            after = postprocess_chinese_cue(
                proposed, self.operations, self.episode_replacements
            )
            if _ASS_TAG_RE.findall(after) != _ASS_TAG_RE.findall(before.chinese):
                staged_entry["status"] = "rejected"
                staged_entry["rejection_reason"] = (
                    f"cue {cue_id} postprocess changes ASS/HTML tags"
                )
                self.history.append(
                    {
                        "kind": "group_rejected",
                        "group_id": group_id,
                        "reason": staged_entry["rejection_reason"],
                    }
                )
                self._commit()
                return {
                    "group_id": group_id,
                    "status": "rejected",
                    "reason": staged_entry["rejection_reason"],
                }
            if after != without_episode:
                staged_entry["status"] = "rejected"
                staged_entry["rejection_reason"] = (
                    f"cue {cue_id} conflicts with authoritative episode replacements"
                )
                self.history.append(
                    {"kind": "group_rejected", "group_id": group_id, "reason": staged_entry["rejection_reason"]}
                )
                self._commit()
                return {"group_id": group_id, "status": "rejected", "reason": staged_entry["rejection_reason"]}
            candidate[cue_id].chinese = after
            changes.append(
                {
                    "id": cue_id,
                    "before": before.chinese,
                    "proposed": proposed,
                    "after_postprocess": after,
                }
            )

        glossary_violations = self._glossary_violations(affected_ids, candidate)
        if glossary_violations:
            staged_entry["status"] = "rejected"
            staged_entry["rejection_reason"] = "glossary violation: " + "; ".join(
                glossary_violations
            )
            self.history.append(
                {
                    "kind": "group_rejected",
                    "group_id": group_id,
                    "reason": staged_entry["rejection_reason"],
                }
            )
            self._commit()
            return {
                "group_id": group_id,
                "status": "rejected",
                "reason": staged_entry["rejection_reason"],
            }

        for cue_id, before in self.current.items():
            after = candidate[cue_id]
            if after.eng != before.eng or after.meta != before.meta:
                raise AssertionError("repair attempted to change English, timeline, tags, or structure")
            if cue_id not in affected_ids and after.chinese != before.chinese:
                raise AssertionError("repair attempted to change a cue outside the group")

        previous_hash = self.current_hash
        self.current = candidate
        staged_entry["status"] = "applied"
        staged_entry["result_artifact_hash"] = self.current_hash
        self.history.append(
            {
                "kind": "group_applied",
                "group_id": group_id,
                "before_artifact_hash": previous_hash,
                "after_artifact_hash": self.current_hash,
                "changes": changes,
            }
        )
        for change in changes:
            if change["proposed"] != change["after_postprocess"]:
                self.history.append(
                    {
                        "kind": "postprocess",
                        "group_id": group_id,
                        "id": change["id"],
                        "before": change["proposed"],
                        "after": change["after_postprocess"],
                    }
                )
        for issue_id in issue_ids:
            self._record_decision(issue_id, "resolved", reason, group_id=group_id)
        invalidated = {
            neighbor
            for cue_id in affected_ids
            for neighbor in range(
                cue_id - self.settings.context_radius,
                cue_id + self.settings.context_radius + 1,
            )
            if neighbor in self.current
        }
        for entry in self.coverage:
            if entry["completed"] is True:
                entry["covered_ids"] = [
                    cue_id
                    for cue_id in entry["covered_ids"]
                    if int(cue_id) not in invalidated
                ]
                entry["invalidated_by_group"] = group_id
        self._commit()
        return {
            "group_id": group_id,
            "status": "applied",
            "artifact_hash": self.current_hash,
            "translations": changes,
            "reinspect_ids": sorted(invalidated),
        }

    def _finish(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        require_exact_fields(dict(arguments), set(), location="finish")
        open_issues = sorted(
            issue_id for issue_id, status in self.issue_states.items() if status == "open"
        )
        pending_groups = [entry["group_id"] for entry in self.staged if entry["status"] == "staged"]
        missing_ids = sorted(set(self.current) - self.covered_ids)
        if self.full_sweeps_completed < 1:
            raise ValueError("finish requires at least one ledger-backed full-episode sweep")
        if missing_ids:
            raise ValueError(f"finish requires complete coverage; missing ids: {missing_ids}")
        if open_issues:
            raise ValueError(f"finish requires every issue to be decided: {open_issues}")
        if pending_groups:
            raise ValueError(f"finish requires no pending staged group: {pending_groups}")
        self.terminal_status = "finish"
        self._commit()
        return {"status": "finish", "artifact_hash": self.current_hash}

    def _escalate(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments), {"reason", "issue_ids"}, location="escalate"
        )
        reason = _nonempty_string(values["reason"], "escalate.reason")
        issue_ids = _string_list(values["issue_ids"], "escalate.issue_ids", nonempty=False)
        unresolved = {
            issue_id for issue_id, status in self.issue_states.items() if status == "open"
        }
        if set(issue_ids) != unresolved:
            raise ValueError("escalate.issue_ids must contain every unresolved open issue")
        for issue_id in issue_ids:
            self._require_issue(issue_id)
            self._record_decision(issue_id, "escalated", reason)
        self.terminal_status = "escalate"
        self.terminal_reason = reason
        self._commit()
        return {"status": "escalate", "reason": reason, "issue_ids": issue_ids}

    def force_escalation(self, reason: str) -> None:
        """Persist every still-open issue as human-review work."""

        for issue_id, status in tuple(self.issue_states.items()):
            if status == "open":
                self._record_decision(issue_id, "escalated", reason)
        self.terminal_status = "escalate"
        self.terminal_reason = reason
        self._commit()

    def _list_resources(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments), {"limit"}, location="list_resources"
        )
        if self.reference_reader is None:
            raise ValueError("reference reader is not configured")
        limit = values["limit"]
        if type(limit) is not int:
            raise ValueError("list_resources.limit must be an integer")
        return [asdict(item) for item in self.reference_reader.list_resources(limit=limit)]

    def _search_subtitles(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments),
            {"query", "paths", "max_results"},
            location="search_subtitles",
        )
        if self.reference_reader is None:
            raise ValueError("reference reader is not configured")
        query = _nonempty_string(values["query"], "search_subtitles.query")
        paths = tuple(_string_list(values["paths"], "search_subtitles.paths", nonempty=False))
        max_results = values["max_results"]
        if type(max_results) is not int:
            raise ValueError("search_subtitles.max_results must be an integer")
        return [
            asdict(item)
            for item in self.reference_reader.search_subtitles(
                query,
                paths=paths or None,
                max_results=max_results,
            )
        ]

    def _read_subtitle_context(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments),
            {"path", "line", "radius"},
            location="read_subtitle_context",
        )
        if self.reference_reader is None:
            raise ValueError("reference reader is not configured")
        path = _nonempty_string(values["path"], "read_subtitle_context.path")
        line = values["line"]
        radius = values["radius"]
        if type(line) is not int or type(radius) is not int:
            raise ValueError("read_subtitle_context line and radius must be integers")
        return asdict(
            self.reference_reader.read_subtitle_context(
                path, line=line, radius=radius
            )
        )

    def _compare_references(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(
            dict(arguments),
            {"left", "right", "max_diff_lines"},
            location="compare",
        )
        if self.reference_reader is None:
            raise ValueError("reference reader is not configured")
        left = _nonempty_string(values["left"], "compare.left")
        right = _nonempty_string(values["right"], "compare.right")
        max_diff_lines = values["max_diff_lines"]
        if type(max_diff_lines) is not int:
            raise ValueError("compare.max_diff_lines must be an integer")
        return asdict(
            self.reference_reader.compare(
                left, right, max_diff_lines=max_diff_lines
            )
        )

    def _webfetch(self, arguments: Mapping[str, JSONValue]) -> JSONValue:
        values = require_exact_fields(dict(arguments), {"url"}, location="webfetch")
        if self.webfetch_executor is None:
            raise ValueError("webfetch executor is not configured")
        url = _nonempty_string(values["url"], "webfetch.url")
        return self.webfetch_executor(url)

    def execute(self, name: str, arguments: Mapping[str, JSONValue]) -> JSONValue:
        """Validate and execute exactly one advertised host action."""

        if self.terminal_status is not None:
            raise ValueError("repair session is already terminal")
        dispatch = {
            "inspect_context": self._inspect_context,
            "merge_issues": self._merge_issues,
            "dismiss_issue": lambda args: self._dismiss_or_keep("dismiss_issue", args),
            "keep_original": lambda args: self._dismiss_or_keep("keep_original", args),
            "open_issue": self._open_issue,
            "stage_group_repair": self._stage_group_repair,
            "finish": self._finish,
            "escalate": self._escalate,
        }
        if self.reference_reader is not None:
            dispatch.update(
                {
                    "list_resources": self._list_resources,
                    "search_subtitles": self._search_subtitles,
                    "read_subtitle_context": self._read_subtitle_context,
                    "compare": self._compare_references,
                }
            )
        if self.webfetch_executor is not None:
            dispatch["webfetch"] = self._webfetch
        try:
            handler = dispatch[name]
        except KeyError as exc:
            raise ValueError(f"unknown repair tool: {name}") from exc
        return handler(arguments)

    def prompt_payload(
        self,
        *,
        effective_glossary: JSONValue,
        episode_memory: JSONValue,
    ) -> dict[str, JSONValue]:
        return {
            "current_artifact_hash": self.current_hash,
            "manifest_hash": self.manifest_hash,
            "suggestions": [asdict(entry) for entry in self.suggestions.values()],  # type: ignore[list-item]
            "issue_states": self.issue_states,
            "decisions": self.decisions,
            "history": self.history,
            "coverage": self.coverage,
            "staged_groups": self.staged,
            "effective_glossary": effective_glossary,
            "episode_memory": episode_memory,
            "budgets": {
                "max_tool_steps": self.settings.max_tool_steps,
                "tool_steps_used": self.tool_steps_used,
                "max_full_sweeps": self.settings.max_full_sweeps,
                "full_sweeps_completed": self.full_sweeps_completed,
                "max_repair_attempts": self.settings.max_repair_attempts,
                "repair_attempts_used": self.repair_attempts_used,
            },
        }


def run_repair_agent(
    *,
    model_settings: RoleModelSettings,
    system_prompt: str,
    session: RepairSession,
    effective_glossary: JSONValue,
    episode_memory: JSONValue,
) -> RepairOutcome:
    """Run one strict JSON-action loop and persist cumulative budgets."""

    if not system_prompt.strip():
        raise ValueError("repair system prompt must be non-empty")
    if not isinstance(effective_glossary, Mapping):
        raise ValueError("effective_glossary must be a JSON object")
    if session.effective_glossary != dict(effective_glossary):
        session.set_effective_glossary(effective_glossary)
    if session.settings.max_full_sweeps <= 0 and session.full_sweeps_completed < 1:
        session.force_escalation("full-sweep budget exhausted")
        return RepairOutcome(
            "escalate",
            session.ordered_current,
            session.current_hash,
            session.state_dir,
            session.tool_steps_used,
            session.repair_attempts_used,
            None,
            session.terminal_reason,
        )
    if session.repair_attempts_used >= session.settings.max_repair_attempts:
        session.force_escalation("repair attempt budget exhausted")
        return RepairOutcome(
            "escalate",
            session.ordered_current,
            session.current_hash,
            session.state_dir,
            session.tool_steps_used,
            session.repair_attempts_used,
            None,
            session.terminal_reason,
        )
    remaining_steps = session.settings.max_tool_steps - session.tool_steps_used
    if remaining_steps <= 0:
        session.force_escalation("tool step budget exhausted")
        return RepairOutcome(
            "escalate",
            session.ordered_current,
            session.current_hash,
            session.state_dir,
            session.tool_steps_used,
            session.repair_attempts_used,
            None,
            session.terminal_reason,
        )

    session.repair_attempts_used += 1
    session._commit()
    model = build_chat_model(model_settings.config)
    messages = (
        ("system", system_prompt),
        (
            "human",
            json.dumps(
                session.prompt_payload(
                    effective_glossary=effective_glossary,
                    episode_memory=episode_memory,
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    )
    steps_before_call = session.tool_steps_used

    def record_exchange(exchange: ToolExchange) -> None:
        session.tool_steps_used = steps_before_call + exchange.step
        session.exchanges.append(_exchange_payload(exchange))
        session._commit()

    try:
        result = invoke_with_tools(
            model,
            messages,
            session.tool_definitions(),
            session.execute,
            max_tool_steps=remaining_steps,
            on_exchange=record_exchange,
        )
    except ToolLoopError as exc:
        session.force_escalation(str(exc))
        return RepairOutcome(
            "escalate",
            session.ordered_current,
            session.current_hash,
            session.state_dir,
            session.tool_steps_used,
            session.repair_attempts_used,
            None,
            str(exc),
        )
    if session.terminal_status is None:
        # A protocol-level final action cannot bypass host finish/escalate gates.
        session.force_escalation(
            "model ended without a successful finish or escalate tool"
        )
    else:
        session._commit()
    return RepairOutcome(
        session.terminal_status,
        session.ordered_current,
        session.current_hash,
        session.state_dir,
        session.tool_steps_used,
        session.repair_attempts_used,
        result,
        session.terminal_reason,
    )
