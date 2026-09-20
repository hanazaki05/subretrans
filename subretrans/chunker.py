"""Split subtitle pairs into model-sized chunks."""

from __future__ import annotations

from collections.abc import Sequence

from .pairs import SubtitlePair
from .utils import estimate_pair_tokens, estimate_pairs_tokens


_TOKEN_SAFETY_MARGIN = 1000


def chunk_pairs(
    pairs: Sequence[SubtitlePair],
    *,
    batch_size: int | None,
    token_soft_limit: int,
    base_prompt_tokens: int,
    model_name: str,
) -> list[list[SubtitlePair]]:
    """Chunk by fixed pair count when ``batch_size`` is set, else by token budget."""

    if not pairs:
        return []
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return chunk_pairs_by_count(pairs, batch_size)

    max_chunk_tokens = token_soft_limit - base_prompt_tokens - _TOKEN_SAFETY_MARGIN
    chunks: list[list[SubtitlePair]] = []
    current: list[SubtitlePair] = []
    current_tokens = 0
    for pair in pairs:
        pair_tokens = estimate_pair_tokens(pair, model_name)
        if current and current_tokens + pair_tokens > max_chunk_tokens:
            chunks.append(current)
            current = [pair]
            current_tokens = pair_tokens
        else:
            current.append(pair)
            current_tokens += pair_tokens
    if current:
        chunks.append(current)
    return chunks


def chunk_pairs_by_count(
    pairs: Sequence[SubtitlePair], pairs_per_chunk: int
) -> list[list[SubtitlePair]]:
    """Split ``pairs`` into consecutive chunks of at most ``pairs_per_chunk``."""

    return [
        list(pairs[start : start + pairs_per_chunk])
        for start in range(0, len(pairs), pairs_per_chunk)
    ]


def chunk_statistics(
    chunks: Sequence[Sequence[SubtitlePair]], model_name: str
) -> dict[str, float]:
    """Summarize chunk sizes in pairs and estimated tokens."""

    if not chunks:
        return {key: 0 for key in (
            "num_chunks", "total_pairs", "avg_pairs_per_chunk", "avg_tokens_per_chunk",
            "min_pairs", "max_pairs", "min_tokens", "max_tokens",
        )}
    pair_counts = [len(chunk) for chunk in chunks]
    token_counts = [estimate_pairs_tokens(chunk, model_name) for chunk in chunks]
    return {
        "num_chunks": len(chunks),
        "total_pairs": sum(pair_counts),
        "avg_pairs_per_chunk": sum(pair_counts) / len(chunks),
        "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
        "min_pairs": min(pair_counts),
        "max_pairs": max(pair_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }


def format_chunk_statistics(
    chunks: Sequence[Sequence[SubtitlePair]], model_name: str
) -> str:
    """One-line human-readable chunk summary for logs."""

    stats = chunk_statistics(chunks, model_name)
    return (
        f"{stats['num_chunks']} chunks, {stats['total_pairs']} pairs; "
        f"pairs/chunk avg={stats['avg_pairs_per_chunk']:.1f} "
        f"range={stats['min_pairs']}-{stats['max_pairs']}; "
        f"tokens/chunk avg={stats['avg_tokens_per_chunk']:.0f} "
        f"range={stats['min_tokens']}-{stats['max_tokens']}"
    )
