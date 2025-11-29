"""
Utility functions for subtitle refinement tool.

Provides token estimation, text processing helpers, and other utilities.
"""

import json
from typing import List, Optional
import tiktoken

from pairs import SubtitlePair


def get_encoding(model_name: str = "gpt-4") -> tiktoken.Encoding:
    """
    Get tiktoken encoding for a specific model.

    Args:
        model_name: Name of the model (e.g., "gpt-4", "gpt-3.5-turbo")

    Returns:
        tiktoken.Encoding object for token counting
    """
    try:
        # Try to get encoding for specific model
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Estimate token count for a given text.

    Args:
        text: Text to estimate tokens for
        model_name: Model name for encoding selection

    Returns:
        Estimated number of tokens
    """
    try:
        encoding = get_encoding(model_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation: ~4 chars per token
        return len(text) // 4


def estimate_pair_tokens(pair: SubtitlePair, model_name: str = "gpt-4") -> int:
    """
    Estimate token count for a subtitle pair in JSON format.

    Args:
        pair: SubtitlePair object
        model_name: Model name for encoding selection

    Returns:
        Estimated number of tokens for this pair when serialized to JSON
    """
    # Convert pair to JSON format that will be sent to LLM
    json_str = json.dumps(pair.to_dict(), ensure_ascii=False)
    return estimate_tokens(json_str, model_name)


def estimate_pairs_tokens(pairs: List[SubtitlePair], model_name: str = "gpt-4") -> int:
    """
    Estimate total token count for a list of subtitle pairs.

    Args:
        pairs: List of SubtitlePair objects
        model_name: Model name for encoding selection

    Returns:
        Estimated total number of tokens
    """
    json_str = json.dumps([p.to_dict() for p in pairs], ensure_ascii=False)
    return estimate_tokens(json_str, model_name)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Extract JSON from LLM response, handling cases where LLM adds extra text.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Extracted JSON string, or None if no valid JSON found
    """
    # Try to find JSON array in the response
    import re

    # Look for JSON array patterns
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    matches = re.findall(array_pattern, response_text, re.DOTALL)

    if matches:
        # Return the longest match (most likely to be complete)
        return max(matches, key=len)

    # If no array found, try to extract from code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # As last resort, return the whole text if it looks like JSON
    stripped = response_text.strip()
    if stripped.startswith('[') and stripped.endswith(']'):
        return stripped

    return None


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to ASS timestamp format (H:MM:SS.CS).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)

    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def parse_timestamp(timestamp: str) -> float:
    """
    Parse ASS timestamp to seconds.

    Args:
        timestamp: Timestamp string in format "H:MM:SS.CS"

    Returns:
        Time in seconds
    """
    import re

    # Parse format like "0:00:01.00"
    pattern = r'(\d+):(\d+):(\d+)\.(\d+)'
    match = re.match(pattern, timestamp)

    if not match:
        return 0.0

    hours, minutes, seconds, centiseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0

    return total_seconds


def clean_whitespace(text: str) -> str:
    """
    Clean excessive whitespace from text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text with normalized whitespace
    """
    import re

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def validate_json_structure(data: any, expected_keys: List[str]) -> bool:
    """
    Validate that JSON data has expected structure.

    Args:
        data: Parsed JSON data
        expected_keys: List of required keys

    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False
        for key in expected_keys:
            if key not in item:
                return False

    return True


def print_verbose_preview(_response_text: str, reasoning_tokens: int, _clear_lines: int = 4) -> None:
    """
    Print reasoning token count.

    Args:
        _response_text: Response text from LLM (kept for compatibility, not used)
        reasoning_tokens: Reasoning tokens reported by API (if available)
        _clear_lines: Number of lines to clear before printing (kept for compatibility, not used)
    """
    print(f"  Reasoning tokens: {reasoning_tokens:,}", flush=True)


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1.23s" or "1m 23s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
