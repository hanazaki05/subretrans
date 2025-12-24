"""
LLM client using OpenAI SDK for subtitle refinement.

This module replaces the HTTP POST approach with OpenAI's official SDK,
while maintaining compatibility with the main project's structure.
"""

import json
import time
import sys
import os
from typing import List, Tuple, Optional, Union, Callable

# OpenAI SDK imports
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_sdk import ConfigSDK, MainModelSettings, TerminologyModelSettings, load_api_key_from_file
from pairs import SubtitlePair
from memory import GlobalMemory, validate_memory_structure
from prompts import (
    build_system_prompt,
    build_user_prompt_for_chunk,
    MEMORY_COMPRESSION_SYSTEM_PROMPT,
    build_memory_compression_prompt,
    validate_response_format
)
from stats import UsageStats
from utils import extract_json_from_response
from serializers import serialize, deserialize, SerializationError


class LLMAPIError(Exception):
    """Exception raised for LLM API errors."""
    pass


def _resolve_model_credentials(
    config: ConfigSDK,
    model_settings: Optional[Union[MainModelSettings, TerminologyModelSettings]] = None,
    verbose: bool = False
) -> Tuple[str, str]:
    """
    Resolve API key and base URL for a specific model.

    Args:
        config: Global ConfigSDK object
        model_settings: Optional model-specific settings
        verbose: Whether to print credential resolution info

    Returns:
        Tuple of (api_key, base_url)
    """
    # Default to global config values
    api_key = config.api_key
    base_url = config.api_base_url

    # Track what overrides were applied
    key_override = False
    url_override = False
    model_name = model_settings.name if model_settings else "unknown"

    # Override with model-specific settings if available
    if model_settings:
        # Check for model-specific base URL
        if model_settings.base_url:
            base_url = model_settings.base_url
            url_override = True

        # Check for model-specific API key file
        if model_settings.key_file:
            # Resolve key file path relative to experiment directory
            key_file_path = model_settings.key_file
            if not os.path.isabs(key_file_path):
                # Relative to experiment directory
                experiment_dir = os.path.dirname(os.path.abspath(__file__))
                key_file_path = os.path.join(experiment_dir, key_file_path)

            try:
                api_key = load_api_key_from_file(key_file_path)
                key_override = True
            except Exception as e:
                raise LLMAPIError(f"Failed to load model-specific API key from {key_file_path}: {str(e)}")

    # Print credential info in verbose mode
    if verbose:
        print(f"\n  [Credential Resolution for {model_name}]")
        if key_override:
            print(f"    API Key: Model-specific ({model_settings.key_file}) [{api_key[:20]}...]")
        else:
            print(f"    API Key: Global (api.key_file) [{api_key[:20]}...]")

        if url_override:
            print(f"    Base URL: Model-specific → {base_url}")
        else:
            print(f"    Base URL: Global (api.base_url) → {base_url}")
        print()

    return api_key, base_url


def call_openai_api_sdk(
    messages: List[dict],
    config: ConfigSDK,
    max_retries: int = 3,
    *,
    model_settings: Optional[Union[MainModelSettings, TerminologyModelSettings]] = None,
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
        max_retries: Maximum number of retry attempts
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
    settings = model_settings or getattr(config, "main_model", None)

    # Resolve credentials (may use model-specific overrides)
    # Show credential info in debug mode (-vvv)
    verbose_creds = getattr(config, "debug_prompts", False)
    api_key, base_url = _resolve_model_credentials(config, settings, verbose=verbose_creds)

    # Initialize OpenAI client with resolved credentials
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=config.api_timeout
    )

    target_model = model_name or (settings.name if settings else getattr(config, "model_name", None))
    target_output_tokens = max_output_tokens or (
        settings.max_output_tokens if settings else getattr(config, "max_output_tokens", None)
    )
    default_reasoning = getattr(settings, "reasoning_effort", None)
    target_reasoning = reasoning_effort if reasoning_effort is not None else default_reasoning
    target_temperature = temperature if temperature is not None else getattr(settings, "temperature", None)

    if not target_model:
        raise LLMAPIError("Model name is not configured")

    if target_output_tokens is None:
        raise LLMAPIError("max_output_tokens must be specified for the selected model")

    attempt = 0
    while attempt < max_retries:
        # Build API call parameters
        api_params = {
            "model": target_model,
            "messages": messages,
            "max_completion_tokens": target_output_tokens
        }

        # Add reasoning effort for GPT-5 models
        if target_reasoning and str(target_model).lower().startswith("gpt-5"):
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
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise LLMAPIError(f"API request timed out after {max_retries} attempts")

            # Check if this is a server error (500+)
            if "status_code" in error_msg or "500" in error_msg or "503" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Server error. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue

            # For other errors, raise immediately
            raise LLMAPIError(f"API request failed: {error_msg}")

        attempt += 1

    raise LLMAPIError(f"Failed after {max_retries} attempts")


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
        # Leverage existing utils.extract_json_from_response()
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from utils import extract_json_from_response
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
    # Build system prompt with memory (using new template-based approach)
    system_content = build_system_prompt(global_memory, config)

    # Serialize pairs using configured format
    pairs_serialized = serialize(pairs_chunk, config.intermediate_format)
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
        response_text, usage = call_openai_api_sdk(
            messages,
            config,
            model_settings=config.main_model
        )

        # Clean response (remove thinking blocks, extract from code blocks)
        cleaned = _clean_llm_response(response_text)

        # For JSON format, additional validation
        if config.intermediate_format.lower() == "json":
            if not validate_response_format(cleaned):
                try:
                    preview = (response_text or "").rstrip()
                except Exception:
                    preview = "[Unavailable raw response]"

                print("\n  [Raw LLM response (invalid format)]:\n")
                print(preview if preview else "[Empty response]")
                print()

                raise LLMAPIError(f"Response is not in expected {config.intermediate_format} format")

        # Deserialize using configured format (with fallback pattern extraction)
        try:
            corrected_pairs = deserialize(cleaned, config.intermediate_format)
        except SerializationError as e:
            # Stage 2: Fallback - try pattern-based extraction
            print(f"\n  [Deserialization failed, attempting pattern extraction...]")

            extracted = _extract_from_format_marker(cleaned, config.intermediate_format)
            if extracted is not None:
                print(f"  [Pattern extraction successful, retrying deserialization...]")
                try:
                    corrected_pairs = deserialize(extracted, config.intermediate_format)
                    print(f"  [Recovery successful!]\n")
                except SerializationError as e2:
                    # Both attempts failed
                    print(f"  [Recovery failed]: {str(e2)}")
                    print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                    print(f"  [Extracted excerpt]: {extracted[:500]}...")
                    raise LLMAPIError(f"Failed to deserialize {config.intermediate_format} response: {str(e)}")
            else:
                # Pattern extraction found nothing
                print(f"  [Pattern extraction found no markers]")
                print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                print(f"  [Raw response excerpt]: {response_text[:500]}...\n")
                raise LLMAPIError(f"Failed to deserialize {config.intermediate_format} response: {str(e)}")

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

        # Verify we got the same number of pairs back
        if len(corrected_pairs) != len(pairs_chunk):
            print(f"  Warning: Expected {len(pairs_chunk)} pairs, got {len(corrected_pairs)}")

        return corrected_pairs, usage, response_text

    except LLMAPIError:
        raise
    except Exception as e:
        raise LLMAPIError(f"Error during chunk refinement: {str(e)}")


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
        response_text, usage = call_openai_api_sdk(
            messages,
            config,
            model_settings=config.main_model
        )

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
        response_text, _ = call_openai_api_sdk(
            messages,
            config,
            max_retries=1,
            model_settings=config.main_model
        )
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
    max_retries: int = 3,
    *,
    model_settings: Optional[Union[MainModelSettings, TerminologyModelSettings]] = None,
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
        max_retries: Maximum number of retry attempts
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
    settings = model_settings or getattr(config, "main_model", None)

    # Resolve credentials (may use model-specific overrides)
    # Show credential info in debug mode (-vvv)
    verbose_creds = getattr(config, "debug_prompts", False)
    api_key, base_url = _resolve_model_credentials(config, settings, verbose=verbose_creds)

    # Initialize OpenAI client with resolved credentials
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=config.api_timeout
    )

    target_model = model_name or (settings.name if settings else getattr(config, "model_name", None))
    target_output_tokens = max_output_tokens or (
        settings.max_output_tokens if settings else getattr(config, "max_output_tokens", None)
    )
    default_reasoning = getattr(settings, "reasoning_effort", None)
    target_reasoning = reasoning_effort if reasoning_effort is not None else default_reasoning
    target_temperature = temperature if temperature is not None else getattr(settings, "temperature", None)

    if not target_model:
        raise LLMAPIError("Model name is not configured")

    if target_output_tokens is None:
        raise LLMAPIError("max_output_tokens must be specified for the selected model")

    attempt = 0
    while attempt < max_retries:
        # Build API call parameters
        api_params = {
            "model": target_model,
            "messages": messages,
            "max_completion_tokens": target_output_tokens,
            "stream": True,  # Enable streaming
            "stream_options": {"include_usage": True}  # Request usage stats in final chunk
        }

        # Add reasoning effort for GPT-5 models
        if target_reasoning and str(target_model).lower().startswith("gpt-5"):
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
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise LLMAPIError(f"API request timed out after {max_retries} attempts")

            # Check if this is a server error (500+)
            if "status_code" in error_msg or "500" in error_msg or "503" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Server error. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue

            # For other errors, raise immediately
            raise LLMAPIError(f"API request failed: {error_msg}")

        attempt += 1

    raise LLMAPIError(f"Failed after {max_retries} attempts")


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
    # Build system prompt with memory (using new template-based approach)
    system_content = build_system_prompt(global_memory, config)

    # Serialize pairs using configured format
    pairs_serialized = serialize(pairs_chunk, config.intermediate_format)
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
            model_settings=config.main_model,
            chunk_callback=chunk_callback
        )

        # Clean response (remove thinking blocks, extract from code blocks)
        cleaned = _clean_llm_response(response_text)

        # For JSON format, additional validation
        if config.intermediate_format.lower() == "json":
            if not validate_response_format(cleaned):
                try:
                    preview = (response_text or "").rstrip()
                except Exception:
                    preview = "[Unavailable raw response]"

                print("\n  [Raw LLM response (invalid format)]:\n")
                print(preview if preview else "[Empty response]")
                print()

                raise LLMAPIError(f"Response is not in expected {config.intermediate_format} format")

        # Deserialize using configured format (with fallback pattern extraction)
        try:
            corrected_pairs = deserialize(cleaned, config.intermediate_format)
        except SerializationError as e:
            # Stage 2: Fallback - try pattern-based extraction
            print(f"\n  [Deserialization failed, attempting pattern extraction...]")

            extracted = _extract_from_format_marker(cleaned, config.intermediate_format)
            if extracted is not None:
                print(f"  [Pattern extraction successful, retrying deserialization...]")
                try:
                    corrected_pairs = deserialize(extracted, config.intermediate_format)
                    print(f"  [Recovery successful!]\n")
                except SerializationError as e2:
                    # Both attempts failed
                    print(f"  [Recovery failed]: {str(e2)}")
                    print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                    print(f"  [Extracted excerpt]: {extracted[:500]}...")
                    raise LLMAPIError(f"Failed to deserialize {config.intermediate_format} response: {str(e)}")
            else:
                # Pattern extraction found nothing
                print(f"  [Pattern extraction found no markers]")
                print(f"  [Cleaned response excerpt]: {cleaned[:500]}...")
                print(f"  [Raw response excerpt]: {response_text[:500]}...\n")
                raise LLMAPIError(f"Failed to deserialize {config.intermediate_format} response: {str(e)}")

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

        # Verify we got the same number of pairs back
        if len(corrected_pairs) != len(pairs_chunk):
            print(f"  Warning: Expected {len(pairs_chunk)} pairs, got {len(corrected_pairs)}")

        return corrected_pairs, usage, response_text

    except LLMAPIError:
        raise
    except Exception as e:
        raise LLMAPIError(f"Error during chunk refinement: {str(e)}")