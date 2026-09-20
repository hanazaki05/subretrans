"""Episode memory: authoritative glossary, learned terminology, and story description."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from langchain_core.language_models import BaseChatModel

from .fsutil import atomic_write_yaml, require_exact_fields
from .pairs import SubtitlePair, pairs_to_json_list
from .prompts import (
    MEMORY_COMPRESSION_SYSTEM_PROMPT,
    build_memory_compression_prompt,
    build_memory_update_system_prompt,
    build_memory_update_user_prompt,
    render_memory_sections,
)
from .providers import clean_response_text, invoke_text
from .stats import UsageStats
from .utils import estimate_tokens

if TYPE_CHECKING:
    from .config import GlossarySettings


logger = logging.getLogger(__name__)

MEMORY_FIELDS = {"user_glossary", "glossary", "story_description"}
_LEGACY_MEMORY_FIELDS = {"style_notes"}
VALID_TERMINOLOGY_TYPES = {
    "person",
    "place",
    "organization",
    "title",
    "acronym",
    "unit",
    "ship",
    "project",
    "law",
    "other",
}
_MAX_EVIDENCE_IDS = 5


@dataclass
class GlobalMemory:
    """Cross-chunk memory carried through serial refinement.

    ``user_glossary`` is authoritative and only ever set from the prompt
    template; ``glossary`` holds model-learned terminology; ``story_description``
    is the cumulative account of the episode so far.
    """

    user_glossary: list[dict[str, Any]] = field(default_factory=list)
    glossary: list[dict[str, Any]] = field(default_factory=list)
    story_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_glossary": self.user_glossary,
            "glossary": self.glossary,
            "story_description": self.story_description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalMemory:
        """Build memory from a validated mapping; a legacy ``style_notes`` key is dropped."""

        if not validate_memory_structure(data):
            raise ValueError("invalid memory structure")
        return cls(
            user_glossary=[dict(entry) for entry in data["user_glossary"]],
            glossary=[dict(entry) for entry in data["glossary"]],
            story_description=data["story_description"],
        )


def validate_memory_structure(payload: Any) -> bool:
    """Check the checkpoint shape; tolerates an old ``style_notes`` string key."""

    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    if not MEMORY_FIELDS <= keys or keys - MEMORY_FIELDS - _LEGACY_MEMORY_FIELDS:
        return False
    if "style_notes" in payload and not isinstance(payload["style_notes"], str):
        return False
    if not isinstance(payload["user_glossary"], list) or not isinstance(payload["glossary"], list):
        return False
    if not isinstance(payload["story_description"], str):
        return False
    for entry in (*payload["user_glossary"], *payload["glossary"]):
        if not isinstance(entry, dict):
            return False
        if not isinstance(entry.get("eng"), str) or not isinstance(entry.get("zh"), str):
            return False
    return True


def load_memory_checkpoint(path: str | os.PathLike[str]) -> GlobalMemory | None:
    """Load a YAML memory checkpoint, or ``None`` when the file does not exist."""

    checkpoint = Path(path)
    if not checkpoint.exists():
        return None
    payload = yaml.safe_load(checkpoint.read_text(encoding="utf-8"))
    if not validate_memory_structure(payload):
        raise ValueError(f"Invalid memory checkpoint: {checkpoint}")
    return GlobalMemory.from_dict(payload)


def save_memory_checkpoint(memory: GlobalMemory, path: str | os.PathLike[str]) -> Path:
    """Atomically persist the complete memory as YAML."""

    return atomic_write_yaml(path, memory.to_dict())


def normalize_term_key(value: Any) -> str:
    """Normalize an English term for matching: NFKC, no zero-width marks, casefold."""

    if not isinstance(value, str):
        return ""
    cleaned = unicodedata.normalize("NFKC", value).replace("\ufeff", "").replace("\u200b", "")
    return re.sub(r"\s+", " ", cleaned.strip()).casefold()


def prune_learned_glossary_against_user_glossary(
    memory: GlobalMemory,
) -> tuple[int, list[dict[str, Any]]]:
    """Drop learned entries whose term is defined by the authoritative glossary."""

    user_keys = {normalize_term_key(entry.get("eng")) for entry in memory.user_glossary}
    user_keys.discard("")
    if not user_keys or not memory.glossary:
        return 0, []
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for entry in memory.glossary:
        key = normalize_term_key(entry.get("eng"))
        (removed if key and key in user_keys else kept).append(entry)
    if removed:
        memory.glossary = kept
    return len(removed), removed


def set_user_glossary(memory: GlobalMemory, entries: list[dict[str, str]]) -> int:
    """Replace the authoritative glossary and prune learned collisions; returns the prune count."""

    memory.user_glossary = [dict(entry) for entry in entries]
    removed_count, _ = prune_learned_glossary_against_user_glossary(memory)
    return removed_count


@dataclass(frozen=True)
class TerminologyEntry:
    """Structured terminology item accepted from the extraction model."""

    eng: str
    zh: str
    type: str
    confidence: float
    evidence_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "eng": self.eng,
            "zh": self.zh,
            "type": self.type,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
        }


def _coerce_evidence_ids(raw_ids: Any) -> tuple[int, ...]:
    if not isinstance(raw_ids, list):
        return ()
    evidence: list[int] = []
    for item in raw_ids:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value not in evidence:
            evidence.append(value)
        if len(evidence) >= _MAX_EVIDENCE_IDS:
            break
    return tuple(evidence)


def _parse_terminology_entries(raw_data: Any, min_confidence: float) -> list[TerminologyEntry]:
    """Keep well-formed candidates at or above the confidence threshold."""

    if not isinstance(raw_data, list):
        return []
    entries: list[TerminologyEntry] = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        eng = str(item.get("eng", "")).strip()
        zh = str(item.get("zh", "")).strip()
        term_type = str(item.get("type", "")).strip().lower()
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        if not eng or not zh or confidence < min_confidence:
            continue
        if term_type not in VALID_TERMINOLOGY_TYPES:
            continue
        entries.append(
            TerminologyEntry(eng, zh, term_type, confidence, _coerce_evidence_ids(item.get("evidence_ids")))
        )
    return entries


def _parse_json_object(text: str, *, location: str) -> dict[str, Any]:
    try:
        payload = json.loads(clean_response_text(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{location} must be a JSON object")
    return payload


def extract_memory_update(
    pairs: list[SubtitlePair],
    previous_story_description: str,
    *,
    model: BaseChatModel,
    settings: GlossarySettings,
    user_glossary: list[dict[str, Any]],
) -> tuple[list[TerminologyEntry], str, UsageStats]:
    """Ask the extraction model for new terminology and the updated story."""

    messages = [
        ("system", build_memory_update_system_prompt(settings.terminology_min_confidence)),
        (
            "human",
            build_memory_update_user_prompt(
                pairs_json=json.dumps(pairs_to_json_list(pairs), ensure_ascii=False, indent=2),
                user_glossary_json=json.dumps(user_glossary, ensure_ascii=False, indent=2),
                previous_story_description=previous_story_description,
            ),
        ),
    ]
    text, usage = invoke_text(model, messages)
    payload = require_exact_fields(
        _parse_json_object(text, location="memory update response"),
        {"glossary", "story_description"},
        location="memory update response",
    )
    if not isinstance(payload["glossary"], list):
        raise ValueError("memory update glossary must be a list")
    if not isinstance(payload["story_description"], str):
        raise ValueError("memory update story_description must be a string")
    entries = _parse_terminology_entries(payload["glossary"], settings.terminology_min_confidence)
    return entries, payload["story_description"].strip(), usage


def update_global_memory(
    memory: GlobalMemory,
    corrected_pairs: list[SubtitlePair],
    *,
    model: BaseChatModel,
    settings: GlossarySettings,
) -> tuple[GlobalMemory, UsageStats]:
    """Merge one chunk's extracted terminology and story into ``memory`` in place.

    The authoritative glossary is locked: candidates whose term it defines are
    skipped whether or not the translation agrees. Learned entries are deduped
    by normalized term and capped at ``settings.max_entries`` (most recent kept).
    """

    if not corrected_pairs:
        return memory, UsageStats()
    prune_learned_glossary_against_user_glossary(memory)
    candidates, story, usage = extract_memory_update(
        corrected_pairs,
        memory.story_description,
        model=model,
        settings=settings,
        user_glossary=memory.user_glossary,
    )
    memory.story_description = story

    user_keys = {normalize_term_key(entry.get("eng")) for entry in memory.user_glossary}
    learned_keys = {normalize_term_key(entry.get("eng")) for entry in memory.glossary}
    added = locked = duplicate = 0
    for candidate in candidates:
        key = normalize_term_key(candidate.eng)
        if key in user_keys:
            locked += 1
            logger.debug("Glossary lock: skipped learned term %r -> %r", candidate.eng, candidate.zh)
            continue
        if key in learned_keys:
            duplicate += 1
            continue
        memory.glossary.append(candidate.to_dict())
        learned_keys.add(key)
        added += 1

    overflow = len(memory.glossary) - settings.max_entries
    if overflow > 0:
        dropped = [entry.get("eng") for entry in memory.glossary[:overflow]]
        memory.glossary = memory.glossary[-settings.max_entries :]
        logger.warning(
            "Learned glossary exceeded %d entries; dropped oldest: %s", settings.max_entries, dropped
        )
    logger.info(
        "Terminology merge: %d candidate(s); added %d, user-locked %d, already learned %d",
        len(candidates),
        added,
        locked,
        duplicate,
    )
    return memory, usage


def estimate_memory_tokens(memory: GlobalMemory, model_name: str) -> int:
    """Estimate the tokens memory adds to the refine prompt, as actually rendered."""

    return estimate_tokens(render_memory_sections(memory), model_name)


def compress_memory(
    memory: GlobalMemory, *, model: BaseChatModel, target_tokens: int
) -> tuple[GlobalMemory, UsageStats]:
    """Compress learned glossary and story with the model; the user glossary is kept verbatim."""

    messages = [
        ("system", MEMORY_COMPRESSION_SYSTEM_PROMPT),
        ("human", build_memory_compression_prompt(memory, target_tokens)),
    ]
    text, usage = invoke_text(model, messages)
    payload = require_exact_fields(
        _parse_json_object(text, location="memory compression response"),
        {"glossary", "story_description"},
        location="memory compression response",
    )
    if not isinstance(payload["glossary"], list) or not all(
        isinstance(entry, dict)
        and isinstance(entry.get("eng"), str)
        and isinstance(entry.get("zh"), str)
        for entry in payload["glossary"]
    ):
        raise ValueError("compressed glossary must be a list of eng/zh entries")
    if not isinstance(payload["story_description"], str):
        raise ValueError("compressed story_description must be a string")
    compressed = GlobalMemory(
        user_glossary=[dict(entry) for entry in memory.user_glossary],
        glossary=[dict(entry) for entry in payload["glossary"]],
        story_description=payload["story_description"].strip(),
    )
    prune_learned_glossary_against_user_glossary(compressed)
    return compressed, usage
