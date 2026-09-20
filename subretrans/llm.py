"""
LLM client using OpenAI SDK for subtitle refinement.

This module replaces the HTTP POST approach with OpenAI's official SDK,
while maintaining compatibility with the main project's structure.
"""

import json
import time
from typing import List, Tuple, Optional, Callable

# OpenAI SDK imports
from langchain_core.messages import AIMessage
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .config import ConfigSDK, RoleModelSettings
from .providers import ModelProtocol, build_chat_model
from .pairs import SubtitlePair
from .memory import (
    GlobalMemory,
    validate_memory_structure,
    prune_learned_glossary_against_user_glossary,
)
from .prompts import (
    build_system_prompt,
    build_user_prompt_for_chunk,
    MEMORY_COMPRESSION_SYSTEM_PROMPT,
    build_memory_compression_prompt,
    validate_response_format
)
from .stats import UsageStats
from .utils import extract_json_from_response
from .serializers import serialize, deserialize, deserialize_best_effort, SerializationError


class LLMAPIError(Exception):
    """Exception raised for LLM API errors."""
    pass


def call_openai_api_sdk(
    messages: List[dict],
    config: ConfigSDK,
    *,
    model_settings: Optional[RoleModelSettings] = None,
    model_name: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None
) -> Tuple[str, UsageStats]:
    """
    Call OpenAI API using official SDK with retry logic.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        config: ConfigSDK object
        model_settings: Optional model settings block
        model_name: Optional explicit model override
        max_output_tokens: Override completion token limit
        reasoning_effort: Override reasoning effort hint (GPT-5 only)
        temperature: Override sampling temperature

    Returns:
        Tuple of (response_text, usage_stats)

    Raises:
        LLMAPIError: If API call fails after retries
    """
    # Determine model settings
    settings = model_settings or config.refine
    provider_config = settings.config
    if provider_config.protocol is not ModelProtocol.OPENAI_CHAT_COMPATIBLE:
        raise LLMAPIError(
            f"chat completions require openai-chat-compatible, got {provider_config.protocol}"
        )

    # Initialize OpenAI client with resolved credentials
    client = OpenAI(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        timeout=provider_config.timeout,
        max_retries=0,
    )

    target_model = model_name or settings.model
    target_output_tokens = max_output_tokens or settings.max_output_tokens
    default_reasoning = settings.reasoning_effort
    target_reasoning = reasoning_effort if reasoning_effort is not None else default_reasoning
    target_temperature = temperature if temperature is not None else settings.temperature

    if not target_model:
        raise LLMAPIError("Model name is not configured")

    if target_output_tokens is None:
        raise LLMAPIError("max_output_tokens must be specified for the selected model")

    retry_limit = settings.max_retries
    attempt = 0
    while attempt <= retry_limit:
        # Build API call parameters
        api_params = {
            "model": target_model,
            "messages": messages,
            "max_completion_tokens": target_output_tokens
        }

        # Add reasoning effort for GPT-5 and Gemini-3 models
        model_lower = str(target_model).lower()
        if target_reasoning and (model_lower.startswith("gpt-5") or model_lower.startswith("gemini-3")):
            api_params["reasoning_effort"] = target_reasoning

        # Add temperature if specified
        if target_temperature is not None:
            api_params["temperature"] = target_temperature

        try:
            # Call OpenAI API using SDK
            response: ChatCompletion = client.chat.completions.create(**api_params)

            # Extract response text
            if not response.choices:
                raise LLMAPIError("No choices in API response")

            response_text = response.choices[0].message.content

            if response_text is None:
                raise LLMAPIError("Response content is None")

            # Extract usage statistics
            usage_data = response.usage
            if usage_data:
                usage_dict = {
                    "prompt_tokens": usage_data.prompt_tokens,
                    "completion_tokens": usage_data.completion_tokens,
                    "total_tokens": usage_data.total_tokens
                }

                # Extract reasoning tokens if available (GPT-5)
                if hasattr(usage_data, "completion_tokens_details") and usage_data.completion_tokens_details:
                    details = usage_data.completion_tokens_details
                    if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                        usage_dict["completion_tokens_details"] = {
                            "reasoning_tokens": details.reasoning_tokens
                        }

                usage = UsageStats.from_api_response(usage_dict)
            else:
                usage = UsageStats()

            return response_text, usage

        except Exception as e:
            error_msg = str(e)

            # Check if this is a timeout error
            if "timeout" in error_msg.lower():
                if attempt < retry_limit:
                    wait_time = 2 ** attempt
                    print(f"  Request timeout. Retrying in {wait_time}s... (retry {attempt + 1}/{retry_limit})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise LLMAPIError(f"API request timed out after {retry_limit + 1} attempts")

            # Check if this is a server error (500+)
            if "status_code" in error_msg or "500" in error_msg or "503" in error_msg:
                if attempt < retry_limit:
                    wait_time = 2 ** attempt
                    print(f"  Server error. Retrying in {wait_time}s... (retry {attempt + 1}/{retry_limit})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue

            # For other errors, raise immediately
            raise LLMAPIError(f"API request failed: {error_msg}")

        attempt += 1

    raise LLMAPIError(f"Failed after {retry_limit + 1} attempts")


def _strip_thinking_blocks(text: str) -> str:
    """
    Remove thinking blocks from LLM response.

    Handles formats like:
    <think>
    **Examining the Task**
    I'm currently focused on...
    </think>

    Args:
        text: Raw text that may contain thinking blocks

    Returns:
        Text with thinking blocks removed
    """
    import re

    # Remove <think>...</think> blocks (case-insensitive, multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


def _extract_from_code_blocks(text: str) -> Optional[str]:
    """
    Extract content from markdown code blocks.

    Handles formats like:
    ```json
    [{"id": 0, ...}]
    ```

    ```xml
    <pair>...</pair>
    ```

    ```toml
    [pair]
    id = 0
    ```

    Also handles code blocks without language specifier:
    ```
    content here
    ```

    Args:
        text: Raw text that may contain code blocks

    Returns:
        Extracted content or None if no code blocks found
    """
    import re

    # Try to find content within code blocks (```...```)
    # Pattern matches both with and without language specifier
    code_block_pattern = r'```(?:\w+)?\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    if matches:
        # Return the first code block content
        return matches[0].strip()

    # No code blocks found
    return None


def _extract_from_format_marker(text: str, format_type: str) -> Optional[str]:
    """
    Extract content starting from format-specific marker.
    Used as fallback when normal deserialization fails.

    IMPORTANT: This function expects text that has ALREADY been cleaned
    by _clean_llm_response() (thinking blocks and code blocks removed).

    Args:
        text: Pre-cleaned text that may have leading commentary
        format_type: One of "json", "xml-pair", "pseudo-toml"

    Returns:
        Extracted content or None if marker not found
    """
    if format_type.lower() == "xml-pair":
        # Find first <pair> tag
        idx = text.find("<pair>")
        if idx != -1:
            return text[idx:].strip()

    elif format_type.lower() == "json":
        return extract_json_from_response(text)

    elif format_type.lower() == "pseudo-toml":
        # Find first [pair] section header
        idx = text.find("[pair]")
        if idx != -1:
            return text[idx:].strip()

    return None


def _detect_duplicate_pairs(pairs: List[SubtitlePair]) -> List[int]:
    """
    Detect duplicate pair IDs in the corrected response.

    Args:
        pairs: List of SubtitlePair objects

    Returns:
        List of duplicate IDs found
    """
    id_counts = {}
    for pair in pairs:
        id_counts[pair.id] = id_counts.get(pair.id, 0) + 1

    duplicates = [id for id, count in id_counts.items() if count > 1]
    return duplicates


def _align_corrected_pair_ids(
    *,
    expected_pairs: List[SubtitlePair],
    corrected_pairs: List[SubtitlePair]
) -> Tuple[List[SubtitlePair], bool]:
    """
    Ensure corrected pair IDs refer to the same items as the input chunk.

    Some models occasionally renumber IDs per-chunk (e.g., 0..N-1) instead of
    preserving the original global IDs. This is particularly dangerous when
    resuming mid-file because applying corrections by ID can overwrite earlier
    pairs in the output.

    Strategy:
    - If all returned IDs are already within the expected ID set: accept.
    - Else, if returned IDs look like local indices (0-based or 1-based),
      remap them to the expected IDs by position.
    - Otherwise: raise to prevent corrupting output.

    Returns:
        (corrected_pairs, remapped) where `remapped` is True if IDs were changed.
    """
    if not expected_pairs or not corrected_pairs:
        return corrected_pairs, False

    expected_ids = [p.id for p in expected_pairs]
    expected_id_set = set(expected_ids)

    returned_ids = [p.id for p in corrected_pairs]
    returned_id_set = set(returned_ids)

    # Fast path: IDs already reference the expected chunk.
    if returned_id_set.issubset(expected_id_set):
        return corrected_pairs, False

    n_expected = len(expected_ids)

    def remap_zero_based() -> bool:
        if not all(0 <= idx < n_expected for idx in returned_id_set):
            return False
        for pair in corrected_pairs:
            pair.id = expected_ids[pair.id]
        return True

    def remap_one_based() -> bool:
        if not all(1 <= idx <= n_expected for idx in returned_id_set):
            return False
        for pair in corrected_pairs:
            pair.id = expected_ids[pair.id - 1]
        return True

    remapped = remap_zero_based() or remap_one_based()
    if remapped:
        # Sanity check: remapped IDs must now reference the expected chunk.
        post_ids = {p.id for p in corrected_pairs}
        if not post_ids.issubset(expected_id_set):
            raise LLMAPIError(
                "Internal error: remapped corrected pair IDs still do not match expected IDs"
            )
        return corrected_pairs, True

    # No safe remapping found: abort to avoid corrupting output.
    expected_min = min(expected_id_set) if expected_id_set else None
    expected_max = max(expected_id_set) if expected_id_set else None
    returned_min = min(returned_id_set) if returned_id_set else None
    returned_max = max(returned_id_set) if returned_id_set else None

    raise LLMAPIError(
        "Corrected pair IDs do not match expected chunk IDs "
        f"(expected IDs ~{expected_min}-{expected_max}, got ~{returned_min}-{returned_max}). "
        "This usually means the model renumbered IDs; refusing to apply to avoid corrupting output."
    )


def _clean_llm_response(text: str) -> str:
    """
    Clean LLM response by removing extraneous content.

    Processing order (CRITICAL):
    1. Remove thinking blocks (<think>...</think>)
    2. Extract from markdown code blocks (```...```)
    3. Return cleaned text

    This handles cases like:
    <think>Analyzing the task...</think>
    ```json
    [{"id": 0, "eng": "Hello", "chinese": "你好"}]
    ```

    Args:
        text: Raw LLM response text

    Returns:
        Cleaned text ready for deserialization
    """
    # Step 1: Remove thinking blocks FIRST
    text = _strip_thinking_blocks(text)

    # Step 2: Extract from code blocks if present
    extracted = _extract_from_code_blocks(text)
    if extracted is not None:
        return extracted

    # No code blocks, return as-is (already stripped of thinking blocks)
    return text.strip()


def refine_chunk_sdk(
    pairs_chunk: List[SubtitlePair],
    global_memory: GlobalMemory,
    config: ConfigSDK,
    print_system_prompt: bool = False
) -> Tuple[List[SubtitlePair], UsageStats, str]:
    """
    Refine a chunk of subtitle pairs using OpenAI SDK.

    Args:
        pairs_chunk: List of SubtitlePair objects to refine
        global_memory: Current global memory
        config: ConfigSDK object
        print_system_prompt: Whether to print system prompt in debug mode (default: False)

    Returns:
        Tuple of (corrected_pairs, usage_stats, response_text)

    Raises:
        LLMAPIError: If refinement fails
    """
    # Keep learned glossary clean before building the prompt
    removed_count, _ = prune_learned_glossary_against_user_glossary(global_memory)
    if removed_count and getattr(config, "verbose", False):
        print(f"  Glossary prune: removed {removed_count} learned entr(y/ies) covered by user glossary (pre-request)")

    # Build system prompt with memory (using new template-based approach)
    system_content = build_system_prompt(global_memory, config)

    # Serialize pairs using configured format
    pairs_serialized = serialize(pairs_chunk, config.intermediate_representation)
    user_content = build_user_prompt_for_chunk(pairs_serialized)

    # Prepare messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Only print system prompt if explicitly requested
    if print_system_prompt and getattr(config, "debug_prompts", False):
        print("\n  System prompt (debug):\n")
        print(system_content.rstrip() if system_content else "[Empty system prompt]")
        print()

    # Call API using SDK
    try:
        response_text, usage = call_role_api_sdk(messages, config, config.refine)

        # Clean response (remove thinking blocks, extract from code blocks)
        cleaned = _clean_llm_response(response_text)

        # For JSON format, additional validation
        if config.intermediate_representation.lower() == "json":
            if not validate_response_format(cleaned):
                try:
                    preview = (response_text or "").rstrip()
                except Exception:
                    preview = "[Unavailable raw response]"

                print("\n  [Raw LLM response (invalid format)]:\n")
                print(preview if preview else "[Empty response]")
                print()

                raise LLMAPIError(f"Response is not in expected {config.intermediate_representation} format")

        # Deserialize using configured format (with fallback pattern extraction)
        try:
            corrected_pairs = deserialize(cleaned, config.intermediate_representation)
        except SerializationError as e:
            # Stage 2: Fallback - try pattern-based extraction
            print(f"\n  [Deserialization failed, attempting pattern extraction...]")

            extracted = _extract_from_format_marker(cleaned, config.intermediate_representation)
            if extracted is not None:
                print(f"  [Pattern extraction successful, retrying deserialization...]")
                try:
                    corrected_pairs = deserialize(extracted, config.intermediate_representation)
                    print(f"  [Recovery successful!]\n")
                except SerializationError as e2:
                    # Stage 3: Best-effort recovery - salvage valid pairs and skip malformed ones
                    recovered_pairs, recovery_errors = deserialize_best_effort(extracted, config.intermediate_representation)
                    if recovered_pairs:
                        skipped = len(recovery_errors)
                        print(f"  [Recovery failed]: {str(e2)}")
                        print(f"  [Best-effort recovery]: salvaged {len(recovered_pairs)} pair(s), skipped {skipped} malformed pair(s)\n")
                        corrected_pairs = recovered_pairs
                    else:
                        # All recovery attempts failed
                        print(f"  [Recovery failed]: {str(e2)}")
                        print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                        print(f"  [Extracted excerpt]: {extracted[:500]}...")
                        raise LLMAPIError(f"Failed to deserialize {config.intermediate_representation} response: {str(e)}")
            else:
                # Pattern extraction found nothing
                recovered_pairs, recovery_errors = deserialize_best_effort(cleaned, config.intermediate_representation)
                if recovered_pairs:
                    skipped = len(recovery_errors)
                    print(f"  [Pattern extraction found no markers]")
                    print(f"  [Best-effort recovery]: salvaged {len(recovered_pairs)} pair(s), skipped {skipped} malformed pair(s)\n")
                    corrected_pairs = recovered_pairs
                else:
                    print(f"  [Pattern extraction found no markers]")
                    print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                    print(f"  [Raw response excerpt]: {response_text[:500]}...\n")
                    raise LLMAPIError(f"Failed to deserialize {config.intermediate_representation} response: {str(e)}")

        # Check for duplicate pairs
        duplicates = _detect_duplicate_pairs(corrected_pairs)
        if duplicates:
            print(f"\n  [Warning]: Duplicate pair IDs detected: {duplicates}")
            print(f"  [Action]: Keeping last occurrence, removing duplicates")

            # Deduplicate: keep last occurrence of each ID
            id_to_pair = {}
            for pair in corrected_pairs:
                id_to_pair[pair.id] = pair  # Last one wins

            # Rebuild list maintaining original order of first appearance
            seen_ids = set()
            deduplicated = []
            for pair in corrected_pairs:
                if pair.id not in seen_ids:
                    deduplicated.append(id_to_pair[pair.id])  # Use last occurrence
                    seen_ids.add(pair.id)

            corrected_pairs = deduplicated
            print(f"  [Result]: {len(deduplicated)} unique pairs retained\n")

        # Align IDs to expected chunk IDs (handles per-chunk renumbering)
        try:
            corrected_pairs, remapped = _align_corrected_pair_ids(
                expected_pairs=pairs_chunk,
                corrected_pairs=corrected_pairs
            )
        except LLMAPIError:
            # Best-effort fallback: drop pairs that do not reference the expected chunk IDs.
            expected_id_set = {p.id for p in pairs_chunk}
            filtered = [p for p in corrected_pairs if p.id in expected_id_set]
            if not filtered:
                raise
            corrected_pairs = filtered
            remapped = False
            print("  [Warning]: Dropped corrected pairs with unexpected IDs (best-effort salvage)")
        if remapped:
            expected_first = pairs_chunk[0].id if pairs_chunk else "?"
            expected_last = pairs_chunk[-1].id if pairs_chunk else "?"
            print(f"  [Warning]: Model returned local IDs; remapped to expected IDs {expected_first}-{expected_last}")

        # Verify we got the same number of pairs back
        if len(corrected_pairs) != len(pairs_chunk):
            expected_ids = [p.id for p in pairs_chunk]
            returned_ids = {p.id for p in corrected_pairs}
            missing_ids = [pid for pid in expected_ids if pid not in returned_ids]
            print(f"  Warning: Expected {len(pairs_chunk)} pairs, got {len(corrected_pairs)}")
            if missing_ids:
                preview = ", ".join(str(i) for i in missing_ids[:20])
                suffix = "..." if len(missing_ids) > 20 else ""
                print(f"  [Warning]: Missing pair IDs: {preview}{suffix}")

        return corrected_pairs, usage, response_text

    except LLMAPIError:
        raise
    except Exception as e:
        raise LLMAPIError(f"Error during chunk refinement: {str(e)}")

def call_role_api_sdk(
    messages: List[dict],
    config: ConfigSDK,
    model_settings: RoleModelSettings,
) -> Tuple[str, UsageStats]:
    """Dispatch one role call according to its configured provider protocol."""

    if model_settings.protocol is ModelProtocol.OPENAI_CHAT_COMPATIBLE:
        return call_openai_api_sdk(
            messages, config, model_settings=model_settings
        )
    if model_settings.protocol is ModelProtocol.OPENAI_RESPONSES:
        return call_openai_api_response(
            messages, config, model_settings=model_settings
        )
    if model_settings.protocol in {
        ModelProtocol.ANTHROPIC_MESSAGES,
        ModelProtocol.GOOGLE_GEMINI,
    }:
        try:
            response = build_chat_model(model_settings.config).invoke(messages)
        except Exception as error:
            raise LLMAPIError(f"API request failed: {error}") from error
        if not isinstance(response, AIMessage):
            raise LLMAPIError("model response must be an AIMessage")
        response_text = response.text
        if not isinstance(response_text, str):
            raise LLMAPIError("AIMessage text must be a string")
        raw_usage = response.usage_metadata or {}
        output_details = raw_usage.get("output_token_details") or {}
        usage = UsageStats(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
            reasoning_tokens=output_details.get("reasoning", 0),
        )
        return response_text, usage
    raise LLMAPIError(f"unsupported role API protocol: {model_settings.protocol}")


def compress_memory_sdk(
    global_memory: GlobalMemory,
    config: ConfigSDK,
    target_tokens: Optional[int] = None
) -> Tuple[GlobalMemory, UsageStats]:
    """
    Compress global memory using OpenAI SDK.

    Args:
        global_memory: Current GlobalMemory to compress
        config: ConfigSDK object
        target_tokens: Target token count

    Returns:
        Tuple of (compressed_memory, usage_stats)

    Raises:
        LLMAPIError: If compression fails
    """
    if target_tokens is None:
        target_tokens = config.memory_token_limit

    # Build prompts
    system_content = MEMORY_COMPRESSION_SYSTEM_PROMPT
    user_content = build_memory_compression_prompt(global_memory, target_tokens)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Call API using SDK
    try:
        response_text, usage = call_role_api_sdk(messages, config, config.refine)

        # Extract JSON
        json_str = extract_json_from_response(response_text)
        if json_str is None:
            json_str = response_text.strip()

        # Parse JSON
        compressed_data = json.loads(json_str)

        # Validate structure
        if not validate_memory_structure(compressed_data):
            raise LLMAPIError("Compressed memory has invalid structure")

        # Create new GlobalMemory from compressed data
        compressed_memory = GlobalMemory.from_dict(compressed_data)

        return compressed_memory, usage

    except json.JSONDecodeError as e:
        raise LLMAPIError(f"Failed to parse memory compression response: {str(e)}")
    except Exception as e:
        raise LLMAPIError(f"Error during memory compression: {str(e)}")


def test_api_connection_sdk(config: ConfigSDK) -> bool:
    """
    Test API connection using OpenAI SDK with a simple request.

    Args:
        config: ConfigSDK object

    Returns:
        True if connection successful, False otherwise
    """
    try:
        messages = [
            {"role": "user", "content": "Reply with just 'OK'"}
        ]
        response_text, _ = call_role_api_sdk(messages, config, config.refine)
        return "OK" in response_text or "ok" in response_text.lower()
    except Exception as e:
        print(f"API connection test failed: {str(e)}")
        return False


# ============================================================================
# STREAMING API FUNCTIONS
# ============================================================================


def call_openai_api_sdk_streaming(
    messages: List[dict],
    config: ConfigSDK,
    *,
    model_settings: Optional[RoleModelSettings] = None,
    model_name: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    chunk_callback: Optional[Callable[[str], None]] = None
) -> Tuple[str, UsageStats]:
    """
    Call OpenAI API using official SDK with STREAMING enabled.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        config: ConfigSDK object
        model_settings: Optional model settings block
        model_name: Optional explicit model override
        max_output_tokens: Override completion token limit
        reasoning_effort: Override reasoning effort hint (GPT-5 only)
        temperature: Override sampling temperature
        chunk_callback: Optional callback function called for each chunk of text

    Returns:
        Tuple of (response_text, usage_stats)

    Raises:
        LLMAPIError: If API call fails after retries
    """
    # Determine model settings
    settings = model_settings or config.refine
    provider_config = settings.config
    if provider_config.protocol is not ModelProtocol.OPENAI_CHAT_COMPATIBLE:
        raise LLMAPIError(
            f"chat completions require openai-chat-compatible, got {provider_config.protocol}"
        )

    # Initialize OpenAI client with resolved credentials
    client = OpenAI(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        timeout=provider_config.timeout,
        max_retries=0,
    )

    target_model = model_name or settings.model
    target_output_tokens = max_output_tokens or settings.max_output_tokens
    default_reasoning = settings.reasoning_effort
    target_reasoning = reasoning_effort if reasoning_effort is not None else default_reasoning
    target_temperature = temperature if temperature is not None else settings.temperature

    if not target_model:
        raise LLMAPIError("Model name is not configured")

    if target_output_tokens is None:
        raise LLMAPIError("max_output_tokens must be specified for the selected model")

    retry_limit = settings.max_retries
    attempt = 0
    while attempt <= retry_limit:
        # Build API call parameters
        api_params = {
            "model": target_model,
            "messages": messages,
            "max_completion_tokens": target_output_tokens,
            "stream": True,  # Enable streaming
            "stream_options": {"include_usage": True}  # Request usage stats in final chunk
        }

        # Add reasoning effort for GPT-5 and Gemini-3 models
        model_lower = str(target_model).lower()
        if target_reasoning and (model_lower.startswith("gpt-5") or model_lower.startswith("gemini-3")):
            api_params["reasoning_effort"] = target_reasoning

        # Add temperature if specified
        if target_temperature is not None:
            api_params["temperature"] = target_temperature

        try:
            # Call OpenAI API using SDK with streaming
            stream = client.chat.completions.create(**api_params)

            # Accumulate response text
            full_response = ""
            usage_dict = {}

            # Process stream chunks
            for chunk in stream:
                # Check if this is a content chunk
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        chunk_text = delta.content
                        full_response += chunk_text

                        # Call callback if provided
                        if chunk_callback:
                            chunk_callback(chunk_text)

                # Check for usage stats in final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_data = chunk.usage
                    usage_dict = {
                        "prompt_tokens": usage_data.prompt_tokens,
                        "completion_tokens": usage_data.completion_tokens,
                        "total_tokens": usage_data.total_tokens
                    }

                    # Extract reasoning tokens if available (GPT-5)
                    if hasattr(usage_data, "completion_tokens_details") and usage_data.completion_tokens_details:
                        details = usage_data.completion_tokens_details
                        if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                            usage_dict["completion_tokens_details"] = {
                                "reasoning_tokens": details.reasoning_tokens
                            }

            # Create usage stats
            if usage_dict:
                usage = UsageStats.from_api_response(usage_dict)
            else:
                usage = UsageStats()

            if not full_response:
                raise LLMAPIError("No content received from streaming API")

            return full_response, usage

        except Exception as e:
            error_msg = str(e)

            # Check if this is a timeout error
            if "timeout" in error_msg.lower():
                if attempt < retry_limit:
                    wait_time = 2 ** attempt
                    print(f"  Request timeout. Retrying in {wait_time}s... (retry {attempt + 1}/{retry_limit})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise LLMAPIError(f"API request timed out after {retry_limit + 1} attempts")

            # Check if this is a server error (500+)
            if "status_code" in error_msg or "500" in error_msg or "503" in error_msg:
                if attempt < retry_limit:
                    wait_time = 2 ** attempt
                    print(f"  Server error. Retrying in {wait_time}s... (retry {attempt + 1}/{retry_limit})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue

            # For other errors, raise immediately
            raise LLMAPIError(f"API request failed: {error_msg}")

        attempt += 1

    raise LLMAPIError(f"Failed after {retry_limit + 1} attempts")


def refine_chunk_sdk_streaming(
    pairs_chunk: List[SubtitlePair],
    global_memory: GlobalMemory,
    config: ConfigSDK,
    chunk_callback: Optional[Callable[[str], None]] = None,
    print_system_prompt: bool = False
) -> Tuple[List[SubtitlePair], UsageStats, str]:
    """
    Refine a chunk of subtitle pairs using OpenAI SDK with STREAMING.

    Args:
        pairs_chunk: List of SubtitlePair objects to refine
        global_memory: Current global memory
        config: ConfigSDK object
        chunk_callback: Optional callback function called for each chunk of streaming text
        print_system_prompt: Whether to print system prompt in debug mode (default: False)

    Returns:
        Tuple of (corrected_pairs, usage_stats, response_text)

    Raises:
        LLMAPIError: If refinement fails
    """
    # Keep learned glossary clean before building the prompt
    removed_count, _ = prune_learned_glossary_against_user_glossary(global_memory)
    if removed_count and getattr(config, "verbose", False):
        print(f"  Glossary prune: removed {removed_count} learned entr(y/ies) covered by user glossary (pre-request)")

    # Build system prompt with memory (using new template-based approach)
    system_content = build_system_prompt(global_memory, config)

    # Serialize pairs using configured format
    pairs_serialized = serialize(pairs_chunk, config.intermediate_representation)
    user_content = build_user_prompt_for_chunk(pairs_serialized)

    # Prepare messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Only print system prompt if explicitly requested (not for streaming real-time output)
    if print_system_prompt and getattr(config, "debug_prompts", False):
        print("\n  System prompt (debug):\n")
        print(system_content.rstrip() if system_content else "[Empty system prompt]")
        print()

    # Call API using SDK with streaming
    try:
        response_text, usage = call_openai_api_sdk_streaming(
            messages,
            config,
            model_settings=config.refine,
            chunk_callback=chunk_callback
        )

        # Clean response (remove thinking blocks, extract from code blocks)
        cleaned = _clean_llm_response(response_text)

        # For JSON format, additional validation
        if config.intermediate_representation.lower() == "json":
            if not validate_response_format(cleaned):
                try:
                    preview = (response_text or "").rstrip()
                except Exception:
                    preview = "[Unavailable raw response]"

                print("\n  [Raw LLM response (invalid format)]:\n")
                print(preview if preview else "[Empty response]")
                print()

                raise LLMAPIError(f"Response is not in expected {config.intermediate_representation} format")

        # Deserialize using configured format (with fallback pattern extraction)
        try:
            corrected_pairs = deserialize(cleaned, config.intermediate_representation)
        except SerializationError as e:
            # Stage 2: Fallback - try pattern-based extraction
            print(f"\n  [Deserialization failed, attempting pattern extraction...]")

            extracted = _extract_from_format_marker(cleaned, config.intermediate_representation)
            if extracted is not None:
                print(f"  [Pattern extraction successful, retrying deserialization...]")
                try:
                    corrected_pairs = deserialize(extracted, config.intermediate_representation)
                    print(f"  [Recovery successful!]\n")
                except SerializationError as e2:
                    # Stage 3: Best-effort recovery - salvage valid pairs and skip malformed ones
                    recovered_pairs, recovery_errors = deserialize_best_effort(extracted, config.intermediate_representation)
                    if recovered_pairs:
                        skipped = len(recovery_errors)
                        print(f"  [Recovery failed]: {str(e2)}")
                        print(f"  [Best-effort recovery]: salvaged {len(recovered_pairs)} pair(s), skipped {skipped} malformed pair(s)\n")
                        corrected_pairs = recovered_pairs
                    else:
                        # All recovery attempts failed
                        print(f"  [Recovery failed]: {str(e2)}")
                        print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                        print(f"  [Extracted excerpt]: {extracted[:500]}...")
                        raise LLMAPIError(f"Failed to deserialize {config.intermediate_representation} response: {str(e)}")
            else:
                # Pattern extraction found nothing
                recovered_pairs, recovery_errors = deserialize_best_effort(cleaned, config.intermediate_representation)
                if recovered_pairs:
                    skipped = len(recovery_errors)
                    print(f"  [Pattern extraction found no markers]")
                    print(f"  [Best-effort recovery]: salvaged {len(recovered_pairs)} pair(s), skipped {skipped} malformed pair(s)\n")
                    corrected_pairs = recovered_pairs
                else:
                    print(f"  [Pattern extraction found no markers]")
                    print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                    print(f"  [Raw response excerpt]: {response_text[:500]}...\n")
                    raise LLMAPIError(f"Failed to deserialize {config.intermediate_representation} response: {str(e)}")

        # Check for duplicate pairs
        duplicates = _detect_duplicate_pairs(corrected_pairs)
        if duplicates:
            print(f"\n  [Warning]: Duplicate pair IDs detected: {duplicates}")
            print(f"  [Action]: Keeping last occurrence, removing duplicates")

            # Deduplicate: keep last occurrence of each ID
            id_to_pair = {}
            for pair in corrected_pairs:
                id_to_pair[pair.id] = pair  # Last one wins

            # Rebuild list maintaining original order of first appearance
            seen_ids = set()
            deduplicated = []
            for pair in corrected_pairs:
                if pair.id not in seen_ids:
                    deduplicated.append(id_to_pair[pair.id])  # Use last occurrence
                    seen_ids.add(pair.id)

            corrected_pairs = deduplicated
            print(f"  [Result]: {len(deduplicated)} unique pairs retained\n")

        # Align IDs to expected chunk IDs (handles per-chunk renumbering)
        try:
            corrected_pairs, remapped = _align_corrected_pair_ids(
                expected_pairs=pairs_chunk,
                corrected_pairs=corrected_pairs
            )
        except LLMAPIError:
            # Best-effort fallback: drop pairs that do not reference the expected chunk IDs.
            expected_id_set = {p.id for p in pairs_chunk}
            filtered = [p for p in corrected_pairs if p.id in expected_id_set]
            if not filtered:
                raise
            corrected_pairs = filtered
            remapped = False
            print("  [Warning]: Dropped corrected pairs with unexpected IDs (best-effort salvage)")
        if remapped:
            expected_first = pairs_chunk[0].id if pairs_chunk else "?"
            expected_last = pairs_chunk[-1].id if pairs_chunk else "?"
            print(f"  [Warning]: Model returned local IDs; remapped to expected IDs {expected_first}-{expected_last}")

        # Verify we got the same number of pairs back
        if len(corrected_pairs) != len(pairs_chunk):
            expected_ids = [p.id for p in pairs_chunk]
            returned_ids = {p.id for p in corrected_pairs}
            missing_ids = [pid for pid in expected_ids if pid not in returned_ids]
            print(f"  Warning: Expected {len(pairs_chunk)} pairs, got {len(corrected_pairs)}")
            if missing_ids:
                preview = ", ".join(str(i) for i in missing_ids[:20])
                suffix = "..." if len(missing_ids) > 20 else ""
                print(f"  [Warning]: Missing pair IDs: {preview}{suffix}")

        return corrected_pairs, usage, response_text

    except LLMAPIError:
        raise
    except Exception as e:
        raise LLMAPIError(f"Error during chunk refinement: {str(e)}")


# ============================================================================
# OPENAI RESPONSES API FUNCTIONS
# ============================================================================


def call_openai_api_response(
    messages: List[dict],
    config: ConfigSDK,
    *,
    model_settings: Optional[RoleModelSettings] = None,
    model_name: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    chunk_callback: Optional[Callable[[str], None]] = None
) -> Tuple[str, UsageStats]:
    """
    Call OpenAI Responses API with streaming (always stream=True).

    The Responses API uses `instructions` for the system prompt and `input`
    for the user message, instead of the messages array.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
                  (will be converted to Responses API format)
        config: ConfigSDK object
        model_settings: Optional model settings block
        model_name: Optional explicit model override
        max_output_tokens: Override completion token limit
        reasoning_effort: Override reasoning effort hint
        temperature: Override sampling temperature
        chunk_callback: Optional callback function called for each chunk of text

    Returns:
        Tuple of (response_text, usage_stats)

    Raises:
        LLMAPIError: If API call fails after retries
    """
    # Determine model settings
    settings = model_settings or config.refine
    provider_config = settings.config
    if provider_config.protocol is not ModelProtocol.OPENAI_RESPONSES:
        raise LLMAPIError(
            f"Responses API requires openai-responses, got {provider_config.protocol}"
        )

    # Initialize OpenAI client
    client = OpenAI(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        timeout=provider_config.timeout,
        max_retries=0,
    )

    target_model = model_name or settings.model
    target_output_tokens = max_output_tokens or settings.max_output_tokens
    default_reasoning = settings.reasoning_effort
    target_reasoning = reasoning_effort if reasoning_effort is not None else default_reasoning
    target_temperature = temperature if temperature is not None else settings.temperature

    if not target_model:
        raise LLMAPIError("Model name is not configured")

    if target_output_tokens is None:
        raise LLMAPIError("max_output_tokens must be specified for the selected model")

    # Convert messages to Responses API format:
    # system messages → instructions, user/assistant messages → input
    instructions = None
    input_messages = []
    for msg in messages:
        if msg["role"] == "system":
            instructions = msg["content"]
        else:
            input_messages.append(msg)

    retry_limit = settings.max_retries
    attempt = 0
    while attempt <= retry_limit:
        # Build API call parameters for Responses API
        api_params = {
            "model": target_model,
            "input": input_messages,
            "max_output_tokens": target_output_tokens,
            "stream": True,  # Response API: always stream
        }

        if instructions:
            api_params["instructions"] = instructions

        # Add reasoning effort for compatible models
        model_lower = str(target_model).lower()
        if target_reasoning and (model_lower.startswith("gpt-5") or model_lower.startswith("gemini-3")):
            api_params["reasoning"] = {"effort": target_reasoning}

        # Add temperature if specified
        if target_temperature is not None:
            api_params["temperature"] = target_temperature

        try:
            # Call OpenAI Responses API with streaming
            stream = client.responses.create(**api_params)

            # Accumulate response text
            full_response = ""
            usage_dict = {}

            # Process stream events
            for event in stream:
                # Handle text delta events
                if event.type == "response.output_text.delta":
                    chunk_text = event.delta
                    full_response += chunk_text
                    if chunk_callback:
                        chunk_callback(chunk_text)

                # Handle completion event with usage
                elif event.type == "response.completed":
                    resp = event.response
                    if hasattr(resp, "usage") and resp.usage:
                        usage_data = resp.usage
                        usage_dict = {
                            "prompt_tokens": getattr(usage_data, "input_tokens", 0),
                            "completion_tokens": getattr(usage_data, "output_tokens", 0),
                            "total_tokens": getattr(usage_data, "total_tokens",
                                                    getattr(usage_data, "input_tokens", 0) +
                                                    getattr(usage_data, "output_tokens", 0))
                        }

                        # Extract reasoning tokens if available
                        if hasattr(usage_data, "output_tokens_details") and usage_data.output_tokens_details:
                            details = usage_data.output_tokens_details
                            if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                                usage_dict["completion_tokens_details"] = {
                                    "reasoning_tokens": details.reasoning_tokens
                                }

            # Create usage stats
            if usage_dict:
                usage = UsageStats.from_api_response(usage_dict)
            else:
                usage = UsageStats()

            if not full_response:
                raise LLMAPIError("No content received from Responses API")

            return full_response, usage

        except Exception as e:
            error_msg = str(e)

            # Check if this is a timeout error
            if "timeout" in error_msg.lower():
                if attempt < retry_limit:
                    wait_time = 2 ** attempt
                    print(f"  Request timeout. Retrying in {wait_time}s... (retry {attempt + 1}/{retry_limit})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise LLMAPIError(f"API request timed out after {retry_limit + 1} attempts")

            # Check if this is a server error (500+)
            if "status_code" in error_msg or "500" in error_msg or "503" in error_msg:
                if attempt < retry_limit:
                    wait_time = 2 ** attempt
                    print(f"  Server error. Retrying in {wait_time}s... (retry {attempt + 1}/{retry_limit})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue

            # For other errors, raise immediately
            raise LLMAPIError(f"Responses API request failed: {error_msg}")

        attempt += 1

    raise LLMAPIError(f"Failed after {retry_limit + 1} attempts")


def refine_chunk_sdk_response(
    pairs_chunk: List[SubtitlePair],
    global_memory: GlobalMemory,
    config: ConfigSDK,
    chunk_callback: Optional[Callable[[str], None]] = None,
    print_system_prompt: bool = False
) -> Tuple[List[SubtitlePair], UsageStats, str]:
    """
    Refine a chunk of subtitle pairs using OpenAI Responses API (always streaming).

    Args:
        pairs_chunk: List of SubtitlePair objects to refine
        global_memory: Current global memory
        config: ConfigSDK object
        chunk_callback: Optional callback function called for each chunk of streaming text
        print_system_prompt: Whether to print system prompt in debug mode (default: False)

    Returns:
        Tuple of (corrected_pairs, usage_stats, response_text)

    Raises:
        LLMAPIError: If refinement fails
    """
    # Keep learned glossary clean before building the prompt
    removed_count, _ = prune_learned_glossary_against_user_glossary(global_memory)
    if removed_count and getattr(config, "verbose", False):
        print(f"  Glossary prune: removed {removed_count} learned entr(y/ies) covered by user glossary (pre-request)")

    # Build system prompt with memory (using new template-based approach)
    system_content = build_system_prompt(global_memory, config)

    # Serialize pairs using configured format
    pairs_serialized = serialize(pairs_chunk, config.intermediate_representation)
    user_content = build_user_prompt_for_chunk(pairs_serialized)

    # Prepare messages (will be converted to Responses API format internally)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Only print system prompt if explicitly requested
    if print_system_prompt and getattr(config, "debug_prompts", False):
        print("\n  System prompt (debug):\n")
        print(system_content.rstrip() if system_content else "[Empty system prompt]")
        print()

    # Call Responses API
    try:
        response_text, usage = call_openai_api_response(
            messages,
            config,
            model_settings=config.refine,
            chunk_callback=chunk_callback
        )

        # Clean response (remove thinking blocks, extract from code blocks)
        cleaned = _clean_llm_response(response_text)

        # For JSON format, additional validation
        if config.intermediate_representation.lower() == "json":
            if not validate_response_format(cleaned):
                try:
                    preview = (response_text or "").rstrip()
                except Exception:
                    preview = "[Unavailable raw response]"

                print("\n  [Raw LLM response (invalid format)]:\n")
                print(preview if preview else "[Empty response]")
                print()

                raise LLMAPIError(f"Response is not in expected {config.intermediate_representation} format")

        # Deserialize using configured format (with fallback pattern extraction)
        try:
            corrected_pairs = deserialize(cleaned, config.intermediate_representation)
        except SerializationError as e:
            # Stage 2: Fallback - try pattern-based extraction
            print(f"\n  [Deserialization failed, attempting pattern extraction...]")

            extracted = _extract_from_format_marker(cleaned, config.intermediate_representation)
            if extracted is not None:
                print(f"  [Pattern extraction successful, retrying deserialization...]")
                try:
                    corrected_pairs = deserialize(extracted, config.intermediate_representation)
                    print(f"  [Recovery successful!]\n")
                except SerializationError as e2:
                    # Stage 3: Best-effort recovery
                    recovered_pairs, recovery_errors = deserialize_best_effort(extracted, config.intermediate_representation)
                    if recovered_pairs:
                        skipped = len(recovery_errors)
                        print(f"  [Recovery failed]: {str(e2)}")
                        print(f"  [Best-effort recovery]: salvaged {len(recovered_pairs)} pair(s), skipped {skipped} malformed pair(s)\n")
                        corrected_pairs = recovered_pairs
                    else:
                        print(f"  [Recovery failed]: {str(e2)}")
                        print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                        print(f"  [Extracted excerpt]: {extracted[:500]}...")
                        raise LLMAPIError(f"Failed to deserialize {config.intermediate_representation} response: {str(e)}")
            else:
                recovered_pairs, recovery_errors = deserialize_best_effort(cleaned, config.intermediate_representation)
                if recovered_pairs:
                    skipped = len(recovery_errors)
                    print(f"  [Pattern extraction found no markers]")
                    print(f"  [Best-effort recovery]: salvaged {len(recovered_pairs)} pair(s), skipped {skipped} malformed pair(s)\n")
                    corrected_pairs = recovered_pairs
                else:
                    print(f"  [Pattern extraction found no markers]")
                    print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                    print(f"  [Raw response excerpt]: {response_text[:500]}...\n")
                    raise LLMAPIError(f"Failed to deserialize {config.intermediate_representation} response: {str(e)}")

        # Check for duplicate pairs
        duplicates = _detect_duplicate_pairs(corrected_pairs)
        if duplicates:
            print(f"\n  [Warning]: Duplicate pair IDs detected: {duplicates}")
            print(f"  [Action]: Keeping last occurrence, removing duplicates")

            id_to_pair = {}
            for pair in corrected_pairs:
                id_to_pair[pair.id] = pair

            seen_ids = set()
            deduplicated = []
            for pair in corrected_pairs:
                if pair.id not in seen_ids:
                    deduplicated.append(id_to_pair[pair.id])
                    seen_ids.add(pair.id)

            corrected_pairs = deduplicated
            print(f"  [Result]: {len(deduplicated)} unique pairs retained\n")

        # Align IDs to expected chunk IDs
        try:
            corrected_pairs, remapped = _align_corrected_pair_ids(
                expected_pairs=pairs_chunk,
                corrected_pairs=corrected_pairs
            )
        except LLMAPIError:
            expected_id_set = {p.id for p in pairs_chunk}
            filtered = [p for p in corrected_pairs if p.id in expected_id_set]
            if not filtered:
                raise
            corrected_pairs = filtered
            remapped = False
            print("  [Warning]: Dropped corrected pairs with unexpected IDs (best-effort salvage)")
        if remapped:
            expected_first = pairs_chunk[0].id if pairs_chunk else "?"
            expected_last = pairs_chunk[-1].id if pairs_chunk else "?"
            print(f"  [Warning]: Model returned local IDs; remapped to expected IDs {expected_first}-{expected_last}")

        # Verify pair count
        if len(corrected_pairs) != len(pairs_chunk):
            expected_ids = [p.id for p in pairs_chunk]
            returned_ids = {p.id for p in corrected_pairs}
            missing_ids = [pid for pid in expected_ids if pid not in returned_ids]
            print(f"  Warning: Expected {len(pairs_chunk)} pairs, got {len(corrected_pairs)}")
            if missing_ids:
                preview = ", ".join(str(i) for i in missing_ids[:20])
                suffix = "..." if len(missing_ids) > 20 else ""
                print(f"  [Warning]: Missing pair IDs: {preview}{suffix}")

        return corrected_pairs, usage, response_text

    except LLMAPIError:
        raise
    except Exception as e:
        raise LLMAPIError(f"Error during chunk refinement: {str(e)}")
