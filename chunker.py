"""
Chunk splitting module for subtitle pairs.

Splits subtitle pairs into manageable chunks that fit within token limits
while preserving logical groupings.
"""

from typing import List
from pairs import SubtitlePair
from utils import estimate_pair_tokens, estimate_pairs_tokens
from config import Config


def chunk_pairs(
    pairs: List[SubtitlePair],
    config: Config,
    base_prompt_tokens: int
) -> List[List[SubtitlePair]]:
    """
    Split subtitle pairs into chunks that fit within token limits or pair count.

    Each chunk will be processed separately by the LLM. The function can chunk
    either by token limits (default) or by fixed pair count (if pairs_per_chunk is set).

    Args:
        pairs: List of all SubtitlePair objects to chunk
        config: Configuration object with token limits and pair count
        base_prompt_tokens: Estimated tokens for system prompt and memory

    Returns:
        List of chunks, where each chunk is a list of SubtitlePair objects
    """
    if not pairs:
        return []

    # If pairs_per_chunk is set, use simple pair-count-based chunking
    if config.pairs_per_chunk is not None and config.pairs_per_chunk > 0:
        return chunk_pairs_by_count(pairs, config.pairs_per_chunk)

    # Otherwise, use token-based chunking (original behavior)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    # Calculate available tokens for pair data
    # Reserve some tokens for output and safety margin
    available_tokens = config.chunk_token_soft_limit - base_prompt_tokens
    safety_margin = 1000  # Reserve for JSON formatting overhead
    max_chunk_tokens = available_tokens - safety_margin

    current_model = config.main_model.name

    for pair in pairs:
        pair_tokens = estimate_pair_tokens(pair, current_model)

        # Check if adding this pair would exceed the limit
        if current_chunk and (current_chunk_tokens + pair_tokens > max_chunk_tokens):
            # Save current chunk and start a new one
            chunks.append(current_chunk)
            current_chunk = [pair]
            current_chunk_tokens = pair_tokens
        else:
            # Add pair to current chunk
            current_chunk.append(pair)
            current_chunk_tokens += pair_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_pairs_by_count(
    pairs: List[SubtitlePair],
    pairs_per_chunk: int
) -> List[List[SubtitlePair]]:
    """
    Split subtitle pairs into chunks by fixed pair count.

    Args:
        pairs: List of all SubtitlePair objects to chunk
        pairs_per_chunk: Number of pairs per chunk

    Returns:
        List of chunks, where each chunk is a list of SubtitlePair objects
    """
    chunks = []
    for i in range(0, len(pairs), pairs_per_chunk):
        chunk = pairs[i:i + pairs_per_chunk]
        chunks.append(chunk)
    return chunks


def estimate_chunk_tokens(chunk: List[SubtitlePair], model_name: str) -> int:
    """
    Estimate total tokens for a chunk of subtitle pairs.

    Args:
        chunk: List of SubtitlePair objects in this chunk
        model_name: Model name for token estimation

    Returns:
        Estimated token count
    """
    return estimate_pairs_tokens(chunk, model_name)


def get_chunk_statistics(chunks: List[List[SubtitlePair]], model_name: str) -> dict:
    """
    Calculate statistics about chunk distribution.

    Args:
        chunks: List of chunks
        model_name: Model name for token estimation

    Returns:
        Dictionary with statistics (num_chunks, total_pairs, tokens_per_chunk, etc.)
    """
    if not chunks:
        return {
            "num_chunks": 0,
            "total_pairs": 0,
            "avg_pairs_per_chunk": 0,
            "avg_tokens_per_chunk": 0,
            "min_pairs": 0,
            "max_pairs": 0,
            "min_tokens": 0,
            "max_tokens": 0
        }

    pair_counts = [len(chunk) for chunk in chunks]
    token_counts = [estimate_chunk_tokens(chunk, model_name) for chunk in chunks]

    return {
        "num_chunks": len(chunks),
        "total_pairs": sum(pair_counts),
        "avg_pairs_per_chunk": sum(pair_counts) / len(chunks),
        "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
        "min_pairs": min(pair_counts),
        "max_pairs": max(pair_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts)
    }


def print_chunk_statistics(chunks: List[List[SubtitlePair]], model_name: str) -> None:
    """
    Print human-readable chunk statistics.

    Args:
        chunks: List of chunks
        model_name: Model name for token estimation
    """
    stats = get_chunk_statistics(chunks, model_name)

    print(f"\n=== Chunk Statistics ===")
    print(f"Number of chunks: {stats['num_chunks']}")
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Average pairs per chunk: {stats['avg_pairs_per_chunk']:.1f}")
    print(f"Pair range: {stats['min_pairs']} - {stats['max_pairs']}")
    print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.0f}")
    print(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"========================\n")


def validate_chunks(
    original_pairs: List[SubtitlePair],
    chunks: List[List[SubtitlePair]]
) -> bool:
    """
    Validate that chunks contain all original pairs exactly once.

    Args:
        original_pairs: Original list of pairs before chunking
        chunks: List of chunks after chunking

    Returns:
        True if validation passes, False otherwise
    """
    # Flatten chunks
    chunked_pairs = []
    for chunk in chunks:
        chunked_pairs.extend(chunk)

    # Check counts match
    if len(chunked_pairs) != len(original_pairs):
        return False

    # Check IDs match (order may differ)
    original_ids = sorted([p.id for p in original_pairs])
    chunked_ids = sorted([p.id for p in chunked_pairs])

    return original_ids == chunked_ids
