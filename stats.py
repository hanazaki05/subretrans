"""
Token usage statistics and cost estimation.

Tracks token usage across LLM calls and estimates costs.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class UsageStats:
    """
    Token usage statistics.

    Attributes:
        prompt_tokens: Total tokens used in prompts
        completion_tokens: Total tokens used in completions
        total_tokens: Total tokens used overall
        reasoning_tokens: Reasoning tokens reported by the API (if available)
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: 'UsageStats') -> 'UsageStats':
        """Add two UsageStats objects together."""
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens
        )

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'UsageStats':
        """Create from dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0)
        )

    @classmethod
    def from_api_response(cls, usage_data: Dict[str, Any]) -> 'UsageStats':
        """
        Create UsageStats from API response usage data.

        Args:
            usage_data: Usage dict from OpenAI API response

        Returns:
            UsageStats object
        """
        return cls(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            reasoning_tokens=usage_data
            .get("completion_tokens_details", {})
            .get("reasoning_tokens", 0)
        )


def init_usage_stats() -> UsageStats:
    """
    Initialize empty usage statistics.

    Returns:
        New UsageStats object with zero counts
    """
    return UsageStats()


def accumulate_usage(total: UsageStats, new_usage: UsageStats) -> UsageStats:
    """
    Accumulate new usage into total.

    Args:
        total: Current total usage
        new_usage: New usage to add

    Returns:
        Updated total usage
    """
    return total + new_usage


def estimate_cost(
    usage: UsageStats,
    price_per_1k_prompt: float,
    price_per_1k_completion: float
) -> float:
    """
    Estimate cost based on usage statistics and pricing.

    Args:
        usage: UsageStats object
        price_per_1k_prompt: Price per 1000 prompt tokens (USD)
        price_per_1k_completion: Price per 1000 completion tokens (USD)

    Returns:
        Estimated cost in USD
    """
    prompt_cost = (usage.prompt_tokens / 1000.0) * price_per_1k_prompt
    completion_cost = (usage.completion_tokens / 1000.0) * price_per_1k_completion
    return prompt_cost + completion_cost


def format_usage_report(usage: UsageStats, cost: float) -> str:
    """
    Format usage statistics as human-readable report.

    Args:
        usage: UsageStats object
        cost: Estimated cost in USD

    Returns:
        Formatted report string
    """
    lines = [
        "\n" + "=" * 50,
        "TOKEN USAGE REPORT",
        "=" * 50,
        f"Prompt tokens:     {usage.prompt_tokens:>10,}",
        f"Completion tokens: {usage.completion_tokens:>10,}",
    ]

    if usage.reasoning_tokens:
        lines.append(f"Reasoning tokens:  {usage.reasoning_tokens:>10,}")

    lines.append(f"Total tokens:      {usage.total_tokens:>10,}")

    lines.extend([
        "-" * 50,
        f"Estimated cost:    ${cost:>10.4f} USD",
        "=" * 50 + "\n"
    ])
    return "\n".join(lines)


def print_usage_report(usage: UsageStats, cost: float) -> None:
    """
    Print usage statistics report to console.

    Args:
        usage: UsageStats object
        cost: Estimated cost in USD
    """
    print(format_usage_report(usage, cost))


def print_chunk_progress(chunk_idx: int, total_chunks: int, chunk_usage: UsageStats) -> None:
    """
    Print progress update for a single chunk.

    Args:
        chunk_idx: Current chunk index (0-based)
        total_chunks: Total number of chunks
        chunk_usage: Usage statistics for this chunk
    """
    progress_pct = ((chunk_idx + 1) / total_chunks) * 100
    print(f"\n[Chunk {chunk_idx + 1}/{total_chunks}] ({progress_pct:.1f}% complete)")
    print(f"  Tokens used: {chunk_usage.total_tokens:,} "
          f"(prompt: {chunk_usage.prompt_tokens:,}, "
          f"completion: {chunk_usage.completion_tokens:,}, "
          f"reasoning: {chunk_usage.reasoning_tokens:,})")


def calculate_token_efficiency(usage: UsageStats) -> Dict[str, float]:
    """
    Calculate token efficiency metrics.

    Args:
        usage: UsageStats object

    Returns:
        Dictionary with efficiency metrics
    """
    if usage.total_tokens == 0:
        return {
            "completion_ratio": 0.0,
            "prompt_ratio": 0.0
        }

    return {
        "completion_ratio": usage.completion_tokens / usage.total_tokens,
        "prompt_ratio": usage.prompt_tokens / usage.total_tokens
    }


def format_token_count(count: int) -> str:
    """
    Format token count with thousand separators.

    Args:
        count: Token count

    Returns:
        Formatted string (e.g., "1,234")
    """
    return f"{count:,}"
