"""
LLM client for OpenAI-style API interactions.

Handles API calls for subtitle refinement and memory compression.
"""

import json
import time
from typing import List, Tuple, Optional, Union
import requests

from config import Config, MainModelSettings, TerminologyModelSettings
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


def call_openai_api(
    messages: List[dict],
    config: Config,
    max_retries: int = 3,
    *,
    model_settings: Optional[Union[MainModelSettings, TerminologyModelSettings]] = None,
    model_name: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None
) -> Tuple[str, UsageStats]:
    """
    Call OpenAI-style API with retry logic.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        config: Configuration object
        max_retries: Maximum number of retry attempts
        model_settings: Optional model settings block (defaults to config.main_model)
        model_name: Optional explicit model override (rare; use model_settings instead)
        max_output_tokens: Override completion token limit for this call
        reasoning_effort: Override reasoning effort hint (GPT-5 only)
        temperature: Override sampling temperature (defaults to model settings)

    Returns:
        Tuple of (response_text, usage_stats)

    Raises:
        LLMAPIError: If API call fails after retries
    """
    url = f"{config.api_base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }

    settings = model_settings or getattr(config, "main_model", None)

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
        payload = {
            "model": target_model,
            "messages": messages,
            "max_completion_tokens": target_output_tokens
        }

        if target_reasoning and str(target_model).lower().startswith("gpt-5"):
            payload["reasoning_effort"] = target_reasoning
        if target_temperature is not None:
            payload["temperature"] = target_temperature

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=getattr(config, "api_timeout", 120),
                # verify=False,
            )
            response.raise_for_status()

            result = response.json()

            # Extract response text
            if "choices" not in result or not result["choices"]:
                raise LLMAPIError("No choices in API response")

            response_text = result["choices"][0]["message"]["content"]

            # Extract usage statistics
            usage_data = result.get("usage", {})
            usage = UsageStats.from_api_response(usage_data)

            return response_text, usage

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                attempt += 1
                continue
            raise LLMAPIError(f"API request timed out after {max_retries} attempts")

        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", None)
            if attempt < max_retries - 1 and status_code is not None and status_code >= 500:
                wait_time = 2 ** attempt
                print(f"  Server error ({status_code}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
                continue
            error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
            raise LLMAPIError(f"API request failed: {error_detail}")

        except Exception as e:
            raise LLMAPIError(f"Unexpected error during API call: {str(e)}")

        # Only reached if no return/raise occurred; move to next attempt
        attempt += 1

    raise LLMAPIError(f"Failed after {max_retries} attempts")


def refine_chunk(
    pairs_chunk: List[SubtitlePair],
    global_memory: GlobalMemory,
    config: Config
) -> Tuple[List[SubtitlePair], UsageStats, str]:
    """
    Refine a chunk of subtitle pairs using LLM.

    Args:
        pairs_chunk: List of SubtitlePair objects to refine
        global_memory: Current global memory
        config: Configuration object

    Returns:
        Tuple of (corrected_pairs, usage_stats, response_text)

    Raises:
        LLMAPIError: If refinement fails
    """
    # Build system prompt with memory
    system_content = build_system_prompt(global_memory)

    # Convert pairs to JSON
    pairs_json = json.dumps(pairs_to_json_list(pairs_chunk), ensure_ascii=False, indent=2)
    user_content = build_user_prompt_for_chunk(pairs_json)

    # Prepare messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    if getattr(config, "debug_prompts", False):
        print("\n  System prompt (debug):\n")
        print(system_content.rstrip() if system_content else "[Empty system prompt]")
        print()

    # Call API
    try:
        response_text, usage = call_openai_api(
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
            # When the model does not return the expected JSON array, print
            # the raw response to help debugging before raising an error.
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


def compress_memory(
    global_memory: GlobalMemory,
    config: Config,
    target_tokens: Optional[int] = None
) -> Tuple[GlobalMemory, UsageStats]:
    """
    Compress global memory using LLM.

    Args:
        global_memory: Current GlobalMemory to compress
        config: Configuration object
        target_tokens: Target token count (defaults to config.memory_token_limit)

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

    # Call API
    try:
        response_text, usage = call_openai_api(
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


def test_api_connection(config: Config) -> bool:
    """
    Test API connection with a simple request.

    Args:
        config: Configuration object

    Returns:
        True if connection successful, False otherwise
    """
    try:
        messages = [
            {"role": "user", "content": "Reply with just 'OK'"}
        ]
        response_text, _ = call_openai_api(
            messages,
            config,
            max_retries=1,
            model_settings=config.main_model
        )
        return "OK" in response_text or "ok" in response_text.lower()
    except Exception as e:
        print(f"API connection test failed: {str(e)}")
        return False
