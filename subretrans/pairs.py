"""Bilingual subtitle pair data structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SubtitlePair:
    """One English-Chinese subtitle pair with the metadata needed to write it back.

    ``id`` is the sequential index within the subtitle file; ``meta`` carries
    timing, style and source line ids so corrections can be applied in place.
    """

    id: int
    eng: str
    chinese: str
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the model-facing representation without metadata."""

        return {"id": self.id, "eng": self.eng, "chinese": self.chinese}

    def __repr__(self) -> str:
        return (
            f"SubtitlePair(id={self.id}, eng={self.eng[:30]!r}, chinese={self.chinese[:30]!r})"
        )


def pairs_to_json_list(pairs: list[SubtitlePair]) -> list[dict[str, Any]]:
    """Convert pairs to the JSON-serializable list sent to models."""

    return [pair.to_dict() for pair in pairs]
