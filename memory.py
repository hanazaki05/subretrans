"""
Global memory management for subtitle refinement.

Maintains terminology, style notes, and context across chunks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json

from config import Config
from pairs import SubtitlePair, pairs_to_json_list
from utils import estimate_tokens, extract_json_from_response


@dataclass
class GlobalMemory:
    """
    Global memory structure for cross-chunk information.

    Attributes:
        glossary: List of terminology entries (people, places, organizations)
        style_notes: Style and tone guidelines
        summary: Brief context or plot summary
    """

    glossary: List[Dict[str, str]] = field(default_factory=list)
    style_notes: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "glossary": self.glossary,
            "style_notes": self.style_notes,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlobalMemory':
        """Create GlobalMemory from dictionary."""
        return cls(
            glossary=data.get("glossary", []),
            style_notes=data.get("style_notes", ""),
            summary=data.get("summary", "")
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def init_global_memory() -> GlobalMemory:
    """
    Initialize empty global memory.

    Returns:
        New GlobalMemory instance
    """
    return GlobalMemory(
        glossary=[],
        style_notes="",
        summary=""
    )


@dataclass
class TerminologyEntry:
    """Structured terminology item returned by the extractor."""

    eng: str
    zh: str
    type: str
    confidence: float
    evidence_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "eng": self.eng,
            "zh": self.zh,
            "type": self.type,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids
        }


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
    "other"
}

MIN_TERM_CONFIDENCE = 0.6




def _coerce_evidence_ids(raw_ids: Any) -> List[int]:
    """Normalize evidence id list coming from LLM output."""
    if not isinstance(raw_ids, list):
        return []

    evidence: List[int] = []
    for item in raw_ids:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx not in evidence:
            evidence.append(idx)
        if len(evidence) >= 5:
            break
    return evidence


def _parse_terminology_entries(raw_data: Any) -> List[TerminologyEntry]:
    """Validate and convert raw JSON array to TerminologyEntry objects."""
    if not isinstance(raw_data, list):
        return []

    entries: List[TerminologyEntry] = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue

        eng = str(item.get("eng", "")).strip()
        zh = str(item.get("zh", "")).strip()
        type_value = str(item.get("type", "")).strip().lower()
        confidence_raw = item.get("confidence")

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            continue

        if not eng or not zh:
            continue
        if confidence < MIN_TERM_CONFIDENCE:
            continue
        if type_value not in VALID_TERMINOLOGY_TYPES:
            continue

        evidence_ids = _coerce_evidence_ids(item.get("evidence_ids", []))
        entries.append(
            TerminologyEntry(
                eng=eng,
                zh=zh,
                type=type_value,
                confidence=confidence,
                evidence_ids=evidence_ids
            )
        )

    return entries


def extract_terminology_from_chunk(
    pairs: List[SubtitlePair],
    config: Config,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    """Extract terminology by calling the dedicated terminology LLM."""
    if not pairs:
        return []

    try:
        from llm_client import call_openai_api, LLMAPIError  # Local import to avoid circular dependency
        from prompts import (
            TERMINOLOGY_EXTRACTION_SYSTEM_PROMPT,
            TERMINOLOGY_EXTRACTION_USER_TEMPLATE
        )
    except Exception:
        # If prompts or client cannot be imported, fail silently to avoid breaking pipeline
        return []

    pairs_json = json.dumps(pairs_to_json_list(pairs), ensure_ascii=False, indent=2)
    user_prompt = TERMINOLOGY_EXTRACTION_USER_TEMPLATE.replace("{{PAIRS_JSON}}", pairs_json)

    messages = [
        {"role": "system", "content": TERMINOLOGY_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response_text, _ = call_openai_api(
            messages,
            config,
            max_retries=max_retries,
            model_settings=getattr(config, "terminology_model", None),
            reasoning_effort=None
        )
    except Exception as e:  # Catch broad exceptions to avoid blocking main flow
        if 'LLMAPIError' in locals() and isinstance(e, LLMAPIError):
            print(f"  Warning: Terminology extraction failed: {e}")
        else:
            print(f"  Warning: Terminology extraction error: {e}")
        return []

    json_str = extract_json_from_response(response_text) or response_text.strip()
    try:
        raw_terms = json.loads(json_str)
    except json.JSONDecodeError as exc:
        print(f"  Warning: Failed to parse terminology response: {exc}")
        return []

    parsed_entries = _parse_terminology_entries(raw_terms)
    return [entry.to_dict() for entry in parsed_entries]


def update_global_memory(
    memory: GlobalMemory,
    corrected_pairs: List[SubtitlePair],
    config: Config
) -> GlobalMemory:
    """
    Update global memory with information from corrected pairs.

    Args:
        memory: Current GlobalMemory object
        corrected_pairs: List of corrected SubtitlePair objects
        config: Runtime configuration (provides API/model info)

    Returns:
        Updated GlobalMemory object
    """
    # Extract new terminology from this chunk
    new_terms = extract_terminology_from_chunk(corrected_pairs, config)

    # Get existing English terms for deduplication
    existing_terms = {entry.get("eng", "") for entry in memory.glossary}

    # Add only new terms
    for term in new_terms:
        if term["eng"] and term["eng"] not in existing_terms:
            memory.glossary.append(term)
            existing_terms.add(term["eng"])

    # Limit glossary size to prevent unbounded growth
    # Keep most recent entries if we exceed limit
    max_glossary_entries = 100
    if len(memory.glossary) > max_glossary_entries:
        memory.glossary = memory.glossary[-max_glossary_entries:]

    return memory


def estimate_memory_tokens(memory: GlobalMemory, model_name: str = "gpt-4") -> int:
    """
    Estimate token count for global memory.

    Args:
        memory: GlobalMemory object
        model_name: Model name for token estimation

    Returns:
        Estimated token count
    """
    # Convert to the format that will be included in prompt
    from prompts import build_memory_section

    memory_text = build_memory_section(memory)
    return estimate_tokens(memory_text, model_name)


def compress_memory_simple(memory: GlobalMemory, max_entries: int = 50) -> GlobalMemory:
    """
    Simple memory compression by limiting glossary size.

    Args:
        memory: GlobalMemory to compress
        max_entries: Maximum number of glossary entries to keep

    Returns:
        Compressed GlobalMemory
    """
    compressed = GlobalMemory(
        glossary=memory.glossary[-max_entries:] if memory.glossary else [],
        style_notes=memory.style_notes[:500] if memory.style_notes else "",  # Truncate to 500 chars
        summary=memory.summary[:500] if memory.summary else ""
    )

    return compressed


def merge_glossary_entries(glossary: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge duplicate glossary entries.

    Args:
        glossary: List of glossary entries

    Returns:
        Deduplicated glossary
    """
    seen = {}
    merged = []

    for entry in glossary:
        eng = entry.get("eng", "")
        if not eng:
            continue

        if eng not in seen:
            seen[eng] = entry
            merged.append(entry)
        else:
            # Update existing entry if new one has more info
            existing = seen[eng]
            if not existing.get("zh") and entry.get("zh"):
                existing["zh"] = entry["zh"]
            if not existing.get("type") and entry.get("type"):
                existing["type"] = entry["type"]

    return merged


def validate_memory_structure(memory_dict: Dict[str, Any]) -> bool:
    """
    Validate that memory dictionary has correct structure.

    Args:
        memory_dict: Dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(memory_dict, dict):
        return False

    # Check required fields exist
    if "glossary" not in memory_dict:
        return False

    # Validate glossary is a list
    if not isinstance(memory_dict["glossary"], list):
        return False

    # Validate glossary entries
    for entry in memory_dict["glossary"]:
        if not isinstance(entry, dict):
            return False
        if "eng" not in entry or "zh" not in entry:
            return False

    return True
