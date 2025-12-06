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
from pairs import SubtitlePair, pairs_to_json_list, pairs_from_json_list
from memory import GlobalMemory, validate_memory_structure
from prompts import (
    build_system_prompt,
    build_user_prompt_for_chunk,
    MEMORY_COMPRESSION_SYSTEM_PROMPT,
    build_memory_compression_prompt,
    validate_response_format
)
from stats import UsageStats
from utils import extract_json_from_response, validate_json_structure


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

    # Convert pairs to JSON
    pairs_json = json.dumps(pairs_to_json_list(pairs_chunk), ensure_ascii=False, indent=2)
    user_content = build_user_prompt_for_chunk(pairs_json)

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

        # Try to extract and parse JSON from response
        json_str = extract_json_from_response(response_text)

        if json_str is None:
            # If extraction failed, try using the whole response
            json_str = response_text.strip()

        # Validate format before parsing
        if not validate_response_format(json_str):
            try:
                preview = (response_text or "").rstrip()
            except Exception:
                preview = "[Unavailable raw response]"

            print("\n  [Raw LLM response (invalid JSON)]:\n")
            print(preview if preview else "[Empty response]")
            print()

            raise LLMAPIError("Response is not in expected JSON format")

        # Parse JSON
        corrected_data = json.loads(json_str)

        # Validate structure
        if not validate_json_structure(corrected_data, ["id", "eng", "chinese"]):
            raise LLMAPIError("Response JSON has invalid structure")

        # Convert back to SubtitlePair objects
        corrected_pairs = pairs_from_json_list(corrected_data)

        # Verify we got the same number of pairs back
        if len(corrected_pairs) != len(pairs_chunk):
            print(f"  Warning: Expected {len(pairs_chunk)} pairs, got {len(corrected_pairs)}")

        return corrected_pairs, usage, response_text

    except json.JSONDecodeError as e:
        raise LLMAPIError(f"Failed to parse LLM response as JSON: {str(e)}\nResponse: {response_text[:500]}...")
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

    # Convert pairs to JSON
    pairs_json = json.dumps(pairs_to_json_list(pairs_chunk), ensure_ascii=False, indent=2)
    user_content = build_user_prompt_for_chunk(pairs_json)

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

        # Try to extract and parse JSON from response
        json_str = extract_json_from_response(response_text)

        if json_str is None:
            # If extraction failed, try using the whole response
            json_str = response_text.strip()

        # Validate format before parsing
        if not validate_response_format(json_str):
            try:
                preview = (response_text or "").rstrip()
            except Exception:
                preview = "[Unavailable raw response]"

            print("\n  [Raw LLM response (invalid JSON)]:\n")
            print(preview if preview else "[Empty response]")
            print()

            raise LLMAPIError("Response is not in expected JSON format")

        # Parse JSON
        corrected_data = json.loads(json_str)

        # Validate structure
        if not validate_json_structure(corrected_data, ["id", "eng", "chinese"]):
            raise LLMAPIError("Response JSON has invalid structure")

        # Convert back to SubtitlePair objects
        corrected_pairs = pairs_from_json_list(corrected_data)

        # Verify we got the same number of pairs back
        if len(corrected_pairs) != len(pairs_chunk):
            print(f"  Warning: Expected {len(pairs_chunk)} pairs, got {len(corrected_pairs)}")

        return corrected_pairs, usage, response_text

    except json.JSONDecodeError as e:
        raise LLMAPIError(f"Failed to parse LLM response as JSON: {str(e)}\nResponse: {response_text[:500]}...")
    except Exception as e:
        raise LLMAPIError(f"Error during chunk refinement: {str(e)}")