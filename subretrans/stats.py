"""Token usage accounting shared by every model role."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class UsageStats:
    """Token counts reported by one or more model calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: UsageStats) -> UsageStats:
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageStats:
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            reasoning_tokens=int(data.get("reasoning_tokens", 0)),
        )


def format_usage_report(
    usage: UsageStats,
    cost: Decimal | None = None,
    *,
    title: str = "TOKEN USAGE REPORT",
) -> str:
    """Render a fixed-width usage summary for terminal output."""

    lines = [
        "=" * 50,
        title,
        "=" * 50,
        f"Prompt tokens:     {usage.prompt_tokens:>10,}",
        f"Completion tokens: {usage.completion_tokens:>10,}",
    ]
    if usage.reasoning_tokens:
        lines.append(f"Reasoning tokens:  {usage.reasoning_tokens:>10,}")
    lines.append(f"Total tokens:      {usage.total_tokens:>10,}")
    lines.append("-" * 50)
    if cost is None:
        lines.append("Estimated cost:    unavailable")
    else:
        lines.append(f"Estimated cost:    ${cost:>10.4f} USD")
    lines.append("=" * 50)
    return "\n".join(lines)
