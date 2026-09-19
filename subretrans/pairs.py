"""
Data structures for subtitle pairs.

Defines SubtitlePair class representing a matched English-Chinese subtitle pair,
along with helper functions for working with subtitle pairs.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class SubtitlePair:
    """
    Represents a pair of English and Chinese subtitles.

    Attributes:
        id: Unique identifier (index in the subtitle sequence)
        eng: English subtitle text (with ASS formatting tags preserved)
        chinese: Chinese subtitle text (with ASS formatting tags preserved)
        meta: Optional metadata (timing, style, etc.)
    """

    id: int
    eng: str
    chinese: str
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SubtitlePair to dictionary format for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "id": self.id,
            "eng": self.eng,
            "chinese": self.chinese
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubtitlePair':
        """
        Create SubtitlePair from dictionary.

        Args:
            data: Dictionary containing at minimum id, eng, and chinese fields

        Returns:
            New SubtitlePair instance
        """
        return cls(
            id=data["id"],
            eng=data["eng"],
            chinese=data["chinese"],
            meta=data.get("meta")
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"SubtitlePair(id={self.id}, eng='{self.eng[:30]}...', chinese='{self.chinese[:30]}...')"


def pairs_to_json_list(pairs: list[SubtitlePair]) -> list[Dict[str, Any]]:
    """
    Convert list of SubtitlePair to JSON-serializable list.

    Args:
        pairs: List of SubtitlePair objects

    Returns:
        List of dictionaries suitable for JSON encoding
    """
    return [pair.to_dict() for pair in pairs]


def pairs_from_json_list(json_list: list[Dict[str, Any]]) -> list[SubtitlePair]:
    """
    Convert JSON list back to SubtitlePair objects.

    Args:
        json_list: List of dictionaries with subtitle pair data

    Returns:
        List of SubtitlePair objects
    """
    return [SubtitlePair.from_dict(item) for item in json_list]


def validate_pair(pair: SubtitlePair) -> bool:
    """
    Validate that a subtitle pair has valid data.

    Args:
        pair: SubtitlePair to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(pair.id, int) or pair.id < 0:
        return False
    if not isinstance(pair.eng, str):
        return False
    if not isinstance(pair.chinese, str):
        return False
    return True


def count_ass_tags(text: str) -> Dict[str, int]:
    """
    Count ASS formatting tags in subtitle text.

    Args:
        text: Subtitle text that may contain ASS tags like {\\i1}, {\\b1}, etc.

    Returns:
        Dictionary mapping tag types to their counts
    """
    import re

    # Match ASS tags like {\i1}, {\b1}, etc.
    tags = re.findall(r'\{[^}]+\}', text)
    tag_counts = {}

    for tag in tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return tag_counts


def verify_tags_preserved(original: str, modified: str) -> bool:
    """
    Verify that ASS tags were preserved between original and modified text.

    Args:
        original: Original subtitle text
        modified: Modified subtitle text

    Returns:
        True if tag counts match (within tolerance), False otherwise
    """
    original_tags = count_ass_tags(original)
    modified_tags = count_ass_tags(modified)

    # Check if all tag types and counts match
    return original_tags == modified_tags
