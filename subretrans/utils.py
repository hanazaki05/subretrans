"""Token estimation and small formatting helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .pairs import SubtitlePair

try:
    import tiktoken
except ImportError:  # pragma: no cover - tiktoken is a hard dependency
    tiktoken = None


def _encoding(model_name: str):
    """Return the tiktoken encoding for ``model_name`` or the cl100k_base fallback."""

    if tiktoken is None:
        raise ImportError("tiktoken is not installed")
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Estimate the token count of ``text``; falls back to four characters per token."""

    try:
        return len(_encoding(model_name).encode(text))
    except Exception:
        return len(text) // 4


def estimate_pair_tokens(pair: SubtitlePair, model_name: str = "gpt-4") -> int:
    """Estimate the tokens one pair occupies when serialized as JSON."""

    return estimate_tokens(json.dumps(pair.to_dict(), ensure_ascii=False), model_name)


def estimate_pairs_tokens(pairs: Sequence[SubtitlePair], model_name: str = "gpt-4") -> int:
    """Estimate the tokens a list of pairs occupies when serialized as JSON."""

    return estimate_tokens(
        json.dumps([pair.to_dict() for pair in pairs], ensure_ascii=False), model_name
    )


def format_time(seconds: float) -> str:
    """Format a duration as ``12.34s`` or ``1m 23.4s``."""

    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {seconds % 60:.1f}s"
